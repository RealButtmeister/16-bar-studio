"""A deterministic, subtractive phrase pass for a completed arrangement.

The input is never changed. Existing performances are either retained, omitted
for a phrase, or shortened at the entrance to an intentional silence. There is
no note generation, retiming, pitch change, or velocity change in this pass.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import math

from .model import Arrangement, BAR, ControlChange, DRUM_ROLES, Note, StudioError

AMOUNTS = ('Off', 'Light', 'Medium', 'Strong')
CORE = frozenset(('kick', 'snare', 'clap'))
LOW = frozenset(('bass', '808', 'sub'))
EFFECTS = frozenset(('fx', 'riser'))
PEDALS = frozenset((64, 66, 69))
PEAKS = frozenset(('hook', 'drop', 'chorus'))
QUIET = frozenset(('intro', 'outro', 'breakdown', 'break'))


@dataclass
class _Voice:
    key: tuple[int, int | None]
    track_id: str
    role: str
    notes: list[Note]
    protected: bool = False

    @property
    def group(self):
        if self.role in LOW:
            return 'low'
        if self.role in DRUM_ROLES:
            return 'drums'
        if self.role in EFFECTS or self.role == 'skip':
            return 'effects'
        return 'music'


def _hash(seed, *parts):
    value = '|'.join(str(part) for part in (seed, *parts))
    return int.from_bytes(hashlib.sha256(value.encode('utf-8')).digest()[:8], 'big')


def _merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def _pedal_windows(track, song_end):
    """Conservative sounding windows, including sostenuto and hold-2.

    Sostenuto may capture fewer notes than sustain. Treating its down window
    as sustain here can preserve an extra phrase, but cannot end a protected
    performance accidentally. CC121 releases these pedal states.
    """
    timed = {event.control for event in track.control_changes}
    state = {control: value for control, value in track.controls
             if control in PEDALS and control not in timed}
    down = 0 if any(value >= 64 for value in state.values()) else None
    result = []
    for event in sorted(track.control_changes, key=lambda event: (event.start, event.order)):
        if event.control == 121:
            state = {}
        elif event.control in PEDALS:
            state[event.control] = event.value
        else:
            continue
        active = any(value >= 64 for value in state.values())
        if active and down is None:
            down = event.start
        elif not active and down is not None:
            if event.start > down:
                result.append((down, event.start))
            down = None
    if down is not None and down < song_end:
        result.append((down, song_end))
    return result


def _sound_end(note, windows):
    index = bisect_right(windows, (note.end, math.inf)) - 1
    if index >= 0 and windows[index][0] <= note.end < windows[index][1]:
        return windows[index][1]
    return note.end


def _active(voice, start, end, windows):
    return any(note.start < end and _sound_end(note, windows) > start
               for note in voice.notes)


def _overlaps(voice, others, start, end, windows_by_track):
    """Do not remove a voice merely because another plays elsewhere nearby."""
    spans = [(max(start, note.start), min(end, _sound_end(note, windows_by_track[voice.key[0]])))
             for note in voice.notes if note.start < end
             and _sound_end(note, windows_by_track[voice.key[0]]) > start]
    other_spans = sorted((max(start, note.start), min(end, _sound_end(note, windows_by_track[other.key[0]])))
                         for other in others for note in other.notes if note.start < end
                         and _sound_end(note, windows_by_track[other.key[0]]) > start)
    if not other_spans:
        return False
    merged = _merge(other_spans)
    ends = [item[1] for item in merged]
    return any((index := bisect_right(ends, left)) < len(merged)
               and merged[index][0] < right for left, right in spans)


def _trim(notes, cuts, end_order):
    if not cuts:
        return list(notes)
    result = []
    starts = [start for start, _ in cuts]
    for note in notes:
        index = bisect_right(starts, note.start) - 1
        if index >= 0 and note.start < cuts[index][1]:
            continue
        following = index + 1
        if following < len(cuts) and note.end > cuts[following][0]:
            result.append(replace(note, duration=cuts[following][0] - note.start,
                                  end_order=end_order))
        else:
            result.append(note)
    return result


def _pedal_cleanup(track, intervals, order, song_end):
    """Release held voices for a silence and restore state before re-entry.

    All unrelated controllers and pitch bends keep their original sequence.
    Only pedal values inside a cut are held off. No channel-wide all-notes-off
    command is introduced. This runs only when no retained voice shares the
    affected channel during the silence.
    """
    intervals = _merge(intervals)
    original = sorted(track.control_changes, key=lambda event: (event.start, event.order))
    timed = {event.control for event in original}
    used = PEDALS & (timed | {control for control, _ in track.controls})
    initial = {control: value for control, value in track.controls
               if control in used and control not in timed}
    output = []
    # The exporter suppresses a static setting once a timed stream exists.
    # Materialize it before adding releases so earlier playback stays intact.
    for control, value in initial.items():
        output.append(ControlChange(0, control, value, order - 10))
    for event in original:
        muted = event.control in used and any(start <= event.start < end
                                              for start, end in intervals)
        output.append(replace(event, value=0) if muted else event)
    for start, end in intervals:
        state = dict(initial)
        for event in original:
            if event.start >= end:
                break
            if event.control == 121:
                state = {}
            elif event.control in used:
                state[event.control] = event.value
        for offset, control in enumerate(sorted(used)):
            output.append(ControlChange(start, control, 0, order + offset))
            # Original events at the return tick retain their relative order.
            if end < song_end:
                output.append(ControlChange(end, control, state.get(control, 0), order + offset))
    track.control_changes = sorted(output, key=lambda event: (event.start, event.order))


def apply_space(arrangement: Arrangement, amount='Medium', protected_ids=()) -> Arrangement:
    """Return a fresh arrangement with whole-phrase supporting-layer rests.

    ``protected_ids`` contains Track IDs; protection includes every drum row
    in a mapped Layer. Kick/snare/clap rows and one recognizable melodic voice
    are also protected automatically. Light uses 8-bar decisions, Medium uses
    4, and Strong uses 2; adjacent rests join into longer passages. Selected
    opening phrases of hooks/drops/choruses retain the full existing stack.

    A shared-channel pedal collision preserves the affected phrase instead
    of altering another instrument's sustain. Decisions describe actual cuts,
    with zero-based start_bar and an exclusive end_bar, like Section.
    """
    if amount not in AMOUNTS:
        raise StudioError('Space must be Off, Light, Medium, or Strong.')
    if isinstance(protected_ids, str):
        protected_ids = (protected_ids,)
    protected_ids = set(protected_ids)
    result = deepcopy(arrangement)
    known_ids = {track.id for track in result.tracks}
    settings = {'kind': 'space_settings', 'amount': amount,
                'protected_ids': sorted(protected_ids & known_ids),
                'auto_protected_ids': [], 'removed_notes': 0,
                'shortened_notes': 0, 'cut_count': 0,
                'pedal_safe_skips': 0, 'added_musical_notes': 0}
    result.decisions.append(settings)
    if amount == 'Off' or not result.sections or not any(track.notes for track in result.tracks):
        return result
    song_end = result.bars * BAR
    phrase_bars = {'Light': 8, 'Medium': 4, 'Strong': 2}[amount]
    settings['phrase_bars'] = phrase_bars
    voices = []
    for index, track in enumerate(result.tracks):
        if track.drum_map:
            by_pitch = defaultdict(list)
            for note in track.notes:
                by_pitch[note.pitch].append(note)
            for pitch, notes in sorted(by_pitch.items()):
                role = track.drum_map.get(pitch, 'percussion')
                voices.append(_Voice((index, pitch), track.id, role, notes,
                                     track.id in protected_ids or role in CORE))
        elif track.notes:
            voices.append(_Voice((index, None), track.id, track.role, list(track.notes),
                                 track.id in protected_ids or track.role in CORE))
    melodic = [voice for voice in voices if voice.group == 'music']
    priority = ('lead', 'vocal', 'pluck', 'arp', 'chords', 'pad', 'stab', 'screech')
    if melodic:
        lead = min(melodic, key=lambda voice: (
            priority.index(voice.role) if voice.role in priority else len(priority),
            -len({note.start // BAR for note in voice.notes}), voice.track_id))
        lead.protected = True
        settings['auto_protected_ids'] = [lead.track_id]
    windows = {index: _pedal_windows(track, song_end)
               for index, track in enumerate(result.tracks)}
    masks = defaultdict(list)
    for section_index, section in enumerate(result.sections):
        for local_bar in range(0, section.bars, phrase_bars):
            start = (section.start_bar + local_bar) * BAR
            end = min(section.end_bar * BAR, start + phrase_bars * BAR)
            if section.kind in PEAKS and local_bar == 0:
                continue
            available = [voice for voice in voices if _active(voice, start, end, windows[voice.key[0]])]
            if len(available) < 2:
                continue
            chosen = {voice.key for voice in available if voice.protected}
            has_core = any(voice.role in CORE for voice in available)
            quiet = section.kind in QUIET or section.energy < .45
            high = section.energy >= .8 and not quiet
            for group in ('music', 'low', 'drums', 'effects'):
                members = [voice for voice in available if voice.group == group]
                if not members:
                    continue
                if group == 'music':
                    keep = (max(2, math.ceil(len(members) * .75)) if amount == 'Light'
                            else 1 if quiet or (amount == 'Strong' and not high) else 2)
                elif group == 'low':
                    keep = 1
                elif group == 'drums':
                    extras = sum(voice.role not in CORE for voice in members)
                    early_verse = (section.kind == 'verse'
                                   and local_bar < min(8, section.bars // 2))
                    extra_keep = (max(1, math.ceil(extras * .75)) if amount == 'Light'
                                  else (0 if has_core and (quiet or early_verse)
                                        else 2 if high else 1) if amount == 'Medium'
                                  else (1 if high else 0))
                    keep = sum(voice.role in CORE for voice in members) + extra_keep
                    if not has_core:
                        keep = max(1, keep)
                else:
                    # Existing one-shot transitions are usually already sparse.
                    keep = max(1, len(members) - (amount != 'Light'))
                selected = [voice for voice in members if voice.key in chosen]
                optional = sorted((voice for voice in members if voice.key not in chosen),
                                  key=lambda voice: (_hash(result.seed, group, voice.track_id, voice.key[1]), voice.key[0]))
                if optional:
                    rotation = (section_index + (start // BAR) // 8) % len(optional)
                    optional = optional[rotation:] + optional[:rotation]
                chosen.update(voice.key for voice in optional[:max(0, keep - len(selected))])
            for voice in available:
                if voice.key in chosen:
                    continue
                companions = [other for other in available if other.key in chosen
                              and other.group == voice.group]
                if not companions:
                    companions = [other for other in available if other.key in chosen]
                if _overlaps(voice, companions, start, end, windows):
                    masks[voice.key].append((start, end))
    masks = {key: _merge(intervals) for key, intervals in masks.items()}
    all_orders = [0]
    for track in result.tracks:
        all_orders.append(track.program_order)
        all_orders.extend(event.order for event in track.control_changes)
        all_orders.extend(event.order for event in track.pitch_bends)
        for note in track.notes:
            all_orders.extend((note.start_order, note.end_order))
    synthetic_order = min(all_orders) - 100
    channel_counts = defaultdict(int)
    for track in result.tracks:
        channel_counts[track.channel] += 1
    # Cancel only a cut whose held tail cannot be released independently.
    # Re-evaluate after cancellations because another row may now remain.
    while True:
        previews = {voice.key: _trim(voice.notes, masks.get(voice.key, []), synthetic_order)
                    for voice in voices}
        releases = defaultdict(list)
        rejected = []
        for voice in voices:
            track_index, pitch = voice.key
            track = result.tracks[track_index]
            if not windows[track_index]:
                continue
            for start, end in masks.get(voice.key, []):
                held = any(note.start < start and _sound_end(note, windows[track_index]) > start
                           for note in previews[voice.key])
                if not held:
                    continue
                others_sound = any(other.key != voice.key and other.key[0] == track_index
                                   and any(note.start < end and _sound_end(note, windows[track_index]) > start
                                           for note in previews[other.key]) for other in voices)
                if channel_counts[track.channel] > 1 or others_sound:
                    rejected.append((voice.key, (start, end)))
                else:
                    releases[track_index].append((start, end))
        if not rejected:
            break
        settings['pedal_safe_skips'] += len(rejected)
        for key, interval in rejected:
            masks[key].remove(interval)
    for track in result.tracks:
        track.notes = []
    for voice in voices:
        track = result.tracks[voice.key[0]]
        cuts = masks.get(voice.key, [])
        track.notes.extend(previews[voice.key])
        removed_counts, shortened_counts = defaultdict(int), defaultdict(int)
        cut_starts = [start for start, _ in cuts]
        for note in voice.notes:
            position = bisect_right(cut_starts, note.start) - 1
            if position >= 0 and note.start < cuts[position][1]:
                removed_counts[position] += 1
            elif position + 1 < len(cuts) and note.end > cuts[position + 1][0]:
                shortened_counts[position + 1] += 1
        for position, (start, end) in enumerate(cuts):
            removed, shortened = removed_counts[position], shortened_counts[position]
            if not removed and not shortened and not any(left == start for left, _ in releases[voice.key[0]]):
                continue
            section = next((section for section in result.sections
                            if section.start_bar * BAR <= start < section.end_bar * BAR), None)
            decision = {'kind': 'space_cut', 'track_id': track.id, 'track_name': track.name,
                        'role': voice.role, 'pitch': voice.key[1],
                        'section': section.name if section else '',
                        'start_bar': start // BAR, 'end_bar': end // BAR,
                        'bars': (end - start) // BAR, 'removed_notes': removed,
                        'shortened_notes': shortened, 'reason': {
                            'music': 'Supporting melody or harmony rests while another voice continues',
                            'low': 'One low-end part rests while the other continues',
                            'drums': 'Supporting percussion rests while the backbone continues',
                            'effects': 'Overlapping effects take turns'}[voice.group]}
            result.decisions.append(decision)
            settings['removed_notes'] += removed
            settings['shortened_notes'] += shortened
            settings['cut_count'] += 1
    for index, track in enumerate(result.tracks):
        track.notes.sort(key=lambda note: (note.start, note.start_order, note.pitch, note.end))
        if releases[index]:
            _pedal_cleanup(track, releases[index], synthetic_order + 20, song_end)
    if settings['pedal_safe_skips']:
        result.warnings.append('Space kept some phrases because their held notes share sustain '
                               'with another retained part. Separate MIDI channels allow those parts to rest independently.')
    cuts = [decision for decision in result.decisions if decision.get('kind') == 'space_cut']
    for decision in result.decisions:
        if decision.get('kind', '').startswith('space_'):
            continue
        start = decision.get('start_bar')
        bars = decision.get('bars')
        if start is None or bars is None:
            continue
        section_cuts = [cut for cut in cuts if cut['start_bar'] < start + bars and cut['end_bar'] > start]
        if section_cuts:
            decision['space_applied'] = True
            decision['exact_source'] = False
            decision['complete_source_peak'] = False
            if 'approach' in decision:
                decision['approach'] = 'Existing arranged performance with supporting-layer phrase rests'
            if any(cut['role'] not in DRUM_ROLES for cut in section_cuts):
                decision['pitched_source_preserved'] = False
            decision['active_tracks'] = [track.id for track in result.tracks
                                         if any(start * BAR <= note.start < (start + bars) * BAR for note in track.notes)]
            decision['phrase_activity'] = {track.id: [bar for bar in range(0, bars, 4)
                if any((start + bar) * BAR <= note.start < (start + min(bar + 4, bars)) * BAR
                       for note in track.notes)] for track in result.tracks if track.id in decision['active_tracks']}
            if 'drum_activity' in decision:
                decision['drum_activity'] = {track.id: {str(pitch): {
                    'name': track.drum_names.get(pitch, role), 'role': role,
                    'hits': sum(note.pitch == pitch and start * BAR <= note.start < (start + bars) * BAR
                                for note in track.notes)} for pitch, role in track.drum_map.items()}
                    for track in result.tracks if track.drum_map}
    return result
