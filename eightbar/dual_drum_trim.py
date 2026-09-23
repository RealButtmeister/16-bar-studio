"""Remove AI-selected drum attacks and their paired releases, preserving hits."""
from collections import defaultdict, deque
from fractions import Fraction


def section_drums(midi, roles, section, song_ppq, block_bars):
    """Return per-repeat event omissions, retained trigger hits and note counts.

    Cuts select note onsets in section-relative quarter-note beats. Notes that
    start before a cut keep their natural tails. Event identities match the
    renderer's raw-track ordinals, even when chord replacement changes indices.
    """
    from .dual_master import _song_scan, _rounded

    drums = []
    for track_index, (records, _) in enumerate(_song_scan(midi)):
        pending = defaultdict(deque)
        for tick, order, message, port, _ in records:
            if message.type == 'note_on' and message.velocity:
                pending[port, message.channel, message.note].append((tick, order, message))
            elif message.type in ('note_on', 'note_off'):
                queue = pending[port, message.channel, message.note]
                if not queue:
                    continue
                start, on_order, original = queue.popleft()
                if roles[track_index] == 'Drums' or original.channel == 9:
                    drums.append((track_index, start, on_order, order, original.note, original.velocity))
    cuts_by_track = defaultdict(list)
    for cut in section.get('drum_cuts', []):
        cuts_by_track[cut['track_index']].append(cut)
    omissions = []
    hits = []
    removed = 0
    line_counts = {}
    for repeat in range(section['repeats']):
        dropped = set()
        for track, start, on, off, pitch, velocity in drums:
            line = line_counts.setdefault((track, pitch), {
                'track_index': track, 'track_name': midi.tracks[track].name,
                'pitch': pitch, 'source_drum_notes': 0, 'drum_notes_trimmed': 0,
                'drum_notes_retained': 0, 'protected': roles[track] == 'Keep'})
            line['source_drum_notes'] += 1
            beat = repeat * block_bars * 4 + Fraction(start, midi.ticks_per_beat)
            trim = roles[track] != 'Keep' and any(
                Fraction(cut['start_beat']) <= beat < Fraction(cut['end_beat'])
                and (not cut['pitches'] or pitch in cut['pitches'])
                for cut in cuts_by_track[track])
            if trim:
                dropped.update((on, off))
                removed += 1
                line['drum_notes_trimmed'] += 1
            else:
                line['drum_notes_retained'] += 1
                hits.append((_rounded(beat * song_ppq), velocity))
        omissions.append(dropped)
    total = len(drums) * section['repeats']
    return omissions, sorted(hits), {'source_drum_notes': total,
                                     'drum_notes_trimmed': removed,
                                     'drum_notes_retained': total - removed,
                                     'line_stats': [line_counts[key] for key in sorted(line_counts)]}
