"""Phrase-sized MIDI clip cuts, applied after the existing velocity envelope.

A cut removes time from a note span. A held note can end at the left edge and
resume at the right edge, keeping its already-shaped velocity and original end.
Controllers and routing are not rewritten; this is MIDI editing, not audio gating.
"""
from collections import defaultdict, deque
from fractions import Fraction


def gate_events(events, cuts, ppq, block_offset_ticks, block_length_ticks):
    """Subtract section-relative cut intervals from one eligible routed lane.

    Caller excludes Drums, Keep, muted and channel-10-containing source tracks.
    Input and output use block-relative (tick, order, MIDI message) records.
    Never add a pitch or extend a source note beyond its original span.
    """
    from .dual_master import _rounded

    stats = {'instrument_notes_removed': 0, 'instrument_notes_shortened': 0,
             'instrument_notes_retriggered': 0, 'instrument_notes_affected': 0,
             'removed_note_beats': 0.0}
    windows = []
    for cut in cuts:
        start = max(0, _rounded(Fraction(str(cut['start_beat'])) * ppq) - block_offset_ticks)
        end = min(block_length_ticks,
                  _rounded(Fraction(str(cut['end_beat'])) * ppq) - block_offset_ticks)
        if end > start:
            windows.append((start, end))
    merged = []
    for start, end in sorted(windows):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    if not merged:
        return events, stats

    records = sorted(events, key=lambda event: (event[0], event[1]))
    pending = defaultdict(deque)
    replaced, additions = set(), []
    removed_ticks = 0
    for index, (tick, order, message) in enumerate(records):
        if message.type == 'note_on' and message.velocity:
            pending[message.channel, message.note].append(index)
        elif message.type in ('note_on', 'note_off'):
            queue = pending[message.channel, message.note]
            if not queue:
                continue
            on_index = queue.popleft()
            start, on_order, original = records[on_index]
            fragments = [(start, tick)]
            for left, right in merged:
                pieces = []
                for a, b in fragments:
                    if right <= a or left >= b:
                        pieces.append((a, b))
                    else:
                        if a < left:
                            pieces.append((a, left))
                        if right < b:
                            pieces.append((right, b))
                fragments = pieces
            if fragments == [(start, tick)]:
                continue
            replaced.update((on_index, index))
            stats['instrument_notes_affected'] += 1
            if not fragments:
                stats['instrument_notes_removed'] += 1
            else:
                stats['instrument_notes_shortened'] += 1
            removed_ticks += tick - start - sum(b - a for a, b in fragments)
            for a, b in fragments:
                stats['instrument_notes_retriggered'] += int(a != start)
                additions.append((a, on_order, original.copy()))
                additions.append((b, order, message.copy()))
    stats['removed_note_beats'] = removed_ticks / ppq
    output = [record for index, record in enumerate(records) if index not in replaced]
    output.extend(additions)
    return sorted(output, key=lambda event: (event[0], event[1])), stats
