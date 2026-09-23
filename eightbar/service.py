"""One generation action: arrange, shape musical velocities, export automation."""
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
import mido
from .model import Arrangement, Source, StudioError, BAR
from .midi_io import write_music, write_automation
from .automation import PROTOCOL, iter_notes, validate_envelopes, DENSITIES
from .frozen_worker import VELOCITY_WORKER_FLAG

def safe_name(name):
    value=re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name).strip(' .')[:85]
    return value or 'Instrument'

def _signature(path):
    doc=mido.MidiFile(path,charset='utf-8')
    rows=[]
    for track in doc.tracks:
        events=[]
        for msg in track:
            row=msg.dict()
            if msg.type=='note_on' and msg.velocity>0:row['velocity']='shaped'
            events.append(row)
        rows.append(events)
    return (doc.ticks_per_beat,rows)

def apply_velocity_shaper(raw,final,arrangement=None):
    if arrangement is not None and any(track.drum_map for track in arrangement.tracks):
        return _shape_layer_drums(raw,final,arrangement)
    script=Path(__file__).with_name('velocity_shaper_original.py')
    options=dict(capture_output=True,text=True,encoding='utf-8',errors='replace',
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=180)
    if getattr(sys,'frozen',False):
        with tempfile.TemporaryDirectory(prefix='eightbar-velocity-') as temporary:
            report_path=Path(temporary)/'report.txt'
            process=subprocess.run([sys.executable,VELOCITY_WORKER_FLAG,
                str(raw),str(final),str(report_path)],**options)
            report=report_path.read_text(encoding='utf-8',errors='replace') if report_path.is_file() else ''
        error=report or process.stderr or process.stdout or 'No velocity worker report was returned.'
    else:
        process=subprocess.run([sys.executable,'-X','utf8',str(script),str(raw),str(final)],**options)
        report=process.stdout
        error=process.stderr or process.stdout
    if process.returncode or not Path(final).is_file():
        raise StudioError('The included velocity shaper failed: '+error[-1800:])
    if _signature(raw)!=_signature(final):
        raise StudioError('Velocity shaping changed something besides note velocities. Export stopped.')
    return report


def _shape_layer_drums(raw,final,arrangement):
    """Give the unchanged shaper named drum parts, then restore the Layer layout.

    The temporary note lanes carry each original hit exactly once. Only their
    resulting velocities are copied back, so channel state, note ordering,
    timing, pitches, and the original combined Layer tracks stay intact.
    """
    document=mido.MidiFile(raw,charset='utf-8')
    if len(document.tracks)!=len(arrangement.tracks)+1:
        raise StudioError('Unexpected track layout while shaping Layer drums.')
    split=mido.MidiFile(type=1,ticks_per_beat=document.ticks_per_beat,charset='utf-8')
    routes=[]
    for track_index,midi_track in enumerate(document.tracks):
        track=arrangement.tracks[track_index-1] if track_index else None
        if track is None or not track.drum_map:
            split.tracks.append(midi_track.copy())
            routes.append([(track_index,index) for index,msg in enumerate(midi_track)
                           if msg.type=='note_on' and msg.velocity>0])
            continue
        groups={pitch:[] for pitch in track.drum_map}
        tick=0
        for index,msg in enumerate(midi_track):
            tick+=msg.time
            if msg.type in ('note_on','note_off'):
                if msg.note not in groups:
                    raise StudioError(f'{track.name}: a drum note has no detected drum lane.')
                groups[msg.note].append((tick,index,msg))
        end=tick
        for pitch,events in sorted(groups.items()):
            if not events:
                continue
            lane=mido.MidiTrack([mido.MetaMessage('track_name',
                name=track.drum_map[pitch].replace('_',' ')+f' - MIDI {pitch}',time=0)])
            previous=0;route=[]
            for when,index,msg in events:
                lane.append(msg.copy(time=when-previous));previous=when
                if msg.type=='note_on' and msg.velocity>0:
                    route.append((track_index,index))
            lane.append(mido.MetaMessage('end_of_track',time=end-previous))
            split.tracks.append(lane);routes.append(route)
    with tempfile.TemporaryDirectory(prefix='layer-velocities-',dir=Path(raw).parent) as temporary:
        before=Path(temporary)/'parts.mid';after=Path(temporary)/'shaped.mid'
        split.save(before)
        report=apply_velocity_shaper(before,after)
        shaped=mido.MidiFile(after,charset='utf-8')
        if len(shaped.tracks)!=len(routes):
            raise StudioError('Velocity shaping changed the detected drum parts.')
        for lane,route in zip(shaped.tracks,routes):
            velocities=[msg.velocity for msg in lane if msg.type=='note_on' and msg.velocity>0]
            if len(velocities)!=len(route):
                raise StudioError('Velocity shaping changed the number of drum hits.')
            for (track_index,index),velocity in zip(route,velocities):
                document.tracks[track_index][index].velocity=velocity
    document.save(final)
    if _signature(raw)!=_signature(final):
        raise StudioError('Layer velocity shaping changed something besides velocities. Export stopped.')
    return 'Layer drums were shaped individually and returned to their original Layer.\n\n'+report

def validate_automation_file(path,arrangement,envelopes,density,name):
    doc=mido.MidiFile(path,charset='utf-8')
    found=[];names=[];instruments=[];end=0
    # Note events live in one named track. A conductor track is also allowed.
    for track in doc.tracks:
        t=0
        for msg in track:
            t+=msg.time
            if msg.is_meta:
                if msg.type=='track_name':names.append(msg.name)
                if msg.type=='instrument_name':instruments.append(msg.name)
            elif msg.type in ('note_on','note_off'):
                found.append((t,msg.type,msg.note,msg.velocity))
            else:raise StudioError('Automation unexpectedly contains a controller or another non-note message.')
        end=max(end,t)
    if doc.ticks_per_beat!=960 or name not in names or name not in instruments:
        raise StudioError('Automation timing or instrument labels failed validation.')
    if end!=arrangement.bars*BAR:raise StudioError('Automation length does not match the song.')
    index=0;count=0
    for note in iter_notes(arrangement,envelopes,density):
        expected=((note.start,'note_on',note.pitch,note.velocity),(note.end,'note_off',note.pitch,0))
        if index+2>len(found) or tuple(found[index:index+2])!=expected:
            raise StudioError('Exported automation notes differ from the preview envelope.')
        count+=1;index+=2
    if index!=len(found):raise StudioError('Automation contains extra note events.')
    return count

def export_project(source:Source, arrangement:Arrangement, envelopes_by_track:dict,
                   density:int, output_root, progress=None, original_arrangement=None):
    progress=progress or (lambda text:None)
    if density not in DENSITIES:raise StudioError('Unsupported automation density.')
    known={track.id for track in arrangement.tracks}
    if set(envelopes_by_track)-known:raise StudioError('Automation refers to an unknown instrument.')
    for envs in envelopes_by_track.values():validate_envelopes(arrangement,envs)
    if not any(t.notes for t in arrangement.tracks):raise StudioError('The arrangement contains no musical notes.')
    ai_decisions=[d for d in arrangement.decisions if str(d.get('kind','')).startswith('ai_')]
    reference_name='Reference arrangement.mid' if ai_decisions else 'Original song.mid'
    if original_arrangement is not None:
        if (original_arrangement.bars!=arrangement.bars or original_arrangement.bpm!=arrangement.bpm
                or [t.id for t in original_arrangement.tracks]!=[t.id for t in arrangement.tracks]):
            raise StudioError('The original and edited arrangements must have matching tracks and timing.')
    root=Path(output_root).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    project=root/f'{safe_name(Path(source.path).stem)}_{stamp}_{uuid.uuid4().hex[:6]}'
    project.mkdir()
    try:
        progress('Arranging your parts…')
        raw=project/'._unshaped.mid';final=project/'Song.mid'
        write_music(raw,arrangement)
        progress('Applying your velocity shaper…')
        report=apply_velocity_shaper(raw,final,arrangement)
        raw.unlink()
        (project/'Velocity shaping.txt').write_text(report,encoding='utf-8')
        if original_arrangement is not None:
            progress('Saving the original arrangement for comparison…')
            original_raw=project/'._original_unshaped.mid'
            write_music(original_raw,original_arrangement)
            apply_velocity_shaper(original_raw,project/reference_name,original_arrangement)
            original_raw.unlink()
        progress('Saving individual instruments…')
        shaped=mido.MidiFile(final,charset='utf-8')
        parts_dir=project/'Instruments';parts_dir.mkdir()
        if len(shaped.tracks)!=len(arrangement.tracks)+1:
            raise StudioError('Unexpected musical track layout while splitting the shaped song.')
        part_files=[]
        for i,(track,midi_track) in enumerate(zip(arrangement.tracks,shaped.tracks[1:])):
            part=mido.MidiFile(type=1,ticks_per_beat=shaped.ticks_per_beat,charset='utf-8')
            part.tracks.extend([shaped.tracks[0],midi_track])
            relative=f'Instruments/{i+1:02d}_{safe_name(track.name)}.mid'
            part.save(project/relative);part_files.append({'id':track.id,'name':track.name,'role':track.role,'file':relative})
        automation_dir=project/'Automation';automation_dir.mkdir()
        automation_files=[]
        for i,track in enumerate(arrangement.tracks):
            if track.id not in envelopes_by_track:continue
            progress('Writing and checking automation: '+track.name)
            envs=envelopes_by_track[track.id]
            relative=f'Automation/{i+1:02d}_{safe_name(track.name)}.mid'
            # Separate files intentionally all use channel 1, so every future
            # receiver instance can use the same default routing contract.
            write_automation(project/relative,track.name,0,iter_notes(arrangement,envs,density),arrangement.bars*BAR,arrangement.bpm)
            count=validate_automation_file(project/relative,arrangement,envs,density,track.name)
            automation_files.append({'track_id':track.id,'instrument':track.name,'file':relative,
                'channel':1,'notes':count,'envelopes':[asdict(e) for e in envs]})
        script=Path(__file__).with_name('velocity_shaper_original.py')
        manifest={'format':'16 Bar Studio project','version':4 if ai_decisions else 3,
            'source':{'path':source.path,'sha256':source.sha256,'bars':source.bars},
            'drum_layers':[{'id':track.id,'name':track.name,
                'map':[{'pitch':pitch,'name':track.drum_names.get(pitch,role),'role':role}
                       for pitch,role in sorted(track.drum_map.items())]}
                for track in arrangement.tracks if track.drum_map],
            'genre':arrangement.genre,'style':arrangement.style,'seed':arrangement.seed,
            'bpm':arrangement.bpm,'bars':arrangement.bars,'sections':[asdict(s) for s in arrangement.sections],
            'music':'Song.mid','instruments':part_files,'automation':automation_files,
            'original_music':reference_name if original_arrangement is not None else None,
            'automation_notes_per_quarter':density,'automation_protocol':PROTOCOL,
            'velocity_shaper':{'automatic':True,'music_only':True,
                'layer_drums_shaped_individually':any(track.drum_map for track in arrangement.tracks),
                'sha256':hashlib.sha256(script.read_bytes()).hexdigest()},
            'decisions':arrangement.decisions,'warnings':list(dict.fromkeys(source.warnings+arrangement.warnings)),
            'validation':{'only_musical_velocities_changed':True,'automation_events_verified':True}}
        (project/'Project.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
        if ai_decisions:
            (project/'AI arrangement.json').write_text(json.dumps(ai_decisions,indent=2,ensure_ascii=False),encoding='utf-8')
            ai=next((d for d in ai_decisions if d.get('kind')=='ai_arrangement'),{})
            lines=['AI arrangement', '', str(ai.get('summary','')), '',
                'Every musical note comes from the imported performance. Existing chord notes may have different pitches.',
                'No new melodies, bass notes, or drum hits were composed.',
                'Phrase rests remove note starts; held note tails may continue through a rest.',
                'AI arrangement.json contains the validated AI decisions.',
                'Reference arrangement.mid is the previous rule-based arrangement on the same timeline, for comparison.']
            source_file=Path(source.path)
            if source_file.is_file() and hashlib.sha256(source_file.read_bytes()).hexdigest()==source.sha256:
                shutil.copyfile(source_file,project/'Original source.mid')
                lines.append('Original source.mid is an unchanged copy of the imported 8- or 16-bar performance.')
            (project/'AI arrangement.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        space_decisions=[d for d in arrangement.decisions if str(d.get('kind','')).startswith('space_')]
        if space_decisions:
            (project/'Space edits.json').write_text(json.dumps(space_decisions,indent=2,ensure_ascii=False),encoding='utf-8')
            settings=next((d for d in reversed(space_decisions) if d.get('kind')=='space_settings'),{})
            lines=['Space: '+str(settings.get('amount','Off')),
                   'Compare Song.mid with Original song.mid at the same starting point.',
                   f"{settings.get('removed_notes',0):,} note starts removed; {settings.get('shortened_notes',0):,} tails shortened at cut boundaries.", '']
            track_names={t.id:t.name for t in arrangement.tracks}
            kept=settings.get('protected_ids',[])+settings.get('auto_protected_ids',[])
            if kept:lines.append('Protected parts: '+', '.join(track_names.get(key,key) for key in dict.fromkeys(kept)))
            lines.append('Core kick, snare and clap parts are protected from this pass.\n')
            for cut in space_decisions:
                if cut.get('kind')!='space_cut':continue
                label=cut.get('track_name',cut.get('track_id','Part'))
                if cut.get('pitch') is not None:label+=' / '+cut.get('role','drum').replace('_',' ')
                lines.append(f"{label}: rest in bars {cut['start_bar']+1}–{cut['end_bar']}.")
            if not settings.get('cut_count'):lines.append('No additional cuts were needed with these settings.')
            if settings.get('pedal_safe_skips'):
                lines.append('\nSome phrases were kept because cutting their pedal-held notes could affect another part.')
            (project/'Space edits.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        (project/'READ ME.txt').write_text(
            'Your arranged song\n\nSong.mid contains the full arrangement, with your velocity shaper already applied.\n'
            +((reference_name+' is '+('the previous rule-based arrangement' if ai_decisions else 'the arrangement before Space cuts')+', with velocity shaping, for comparison.\n') if original_arrangement is not None else '')+
            ('AI arrangement.txt and AI arrangement.json describe the AI decisions and any chord adjustments.\n' if ai_decisions else '')+
            ('' if ai_decisions else 'Space edits.txt lists phrase removals and protection choices; Space edits.json contains the same decisions as data.\n')+
            'Layer drums stay in their original Layer tracks; empty drum reference lanes are omitted.\n'
            'Every musical drum hit comes from your source performance; no new hits are composed.\n'
            'Instruments contains the same shaped parts as separate MIDI files.\n'
            'Automation contains velocity-control MIDI for Velocity Pass, one file per selected instrument.\n'
            'Import every file at the same song start and use the BPM in Project.json.\n'
            'Route automation to the control plugin separately from the musical instrument.\n'
            'Velocity Pass uses note-on velocity for the audio level; note-offs hold it.\n'
            'Match the MIDI Out port to the effect input port and keep the FL effect slot Mix at 100%.\n'
            'Automation pitches remain fixed within each section; velocity shapes are independent.\n'
            'Automation keeps following song time during musical gaps, ready for each reentry.\n'
            'The musical velocity shaper has not touched the automation.\n',encoding='utf-8')
        progress('Ready. Music and automation checked.')
        return project
    except Exception as exc:
        (project/'EXPORT FAILED.txt').write_text(str(exc),encoding='utf-8')
        raise
