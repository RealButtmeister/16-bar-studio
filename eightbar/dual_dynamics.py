"""Musical envelopes driven by the imported performance's actual drum hits.

All timestamps use the current pattern's PPQ. This module processes one pitched
instrument; the caller keeps protected drum and Keep lanes out of this path.
"""
from __future__ import annotations

import bisect
import math

import mido


def _parameters(dynamics, ppq, section_length_ticks):
    if type(ppq) is not int or ppq <= 0 or section_length_ticks <= 0:
        raise ValueError('Dynamics require positive musical timing.')
    start = float(dynamics['velocity_start'])
    end = float(dynamics['velocity_end'])
    reaction = float(dynamics['drum_reaction'])
    recovery = float(dynamics['recovery_beats']) * ppq
    if not all(math.isfinite(value) for value in (start, end, reaction, recovery)):
        raise ValueError('Dynamics values must be finite.')
    if not (1 <= start <= 127 and 1 <= end <= 127 and 0 <= reaction <= 0.8
            and ppq * 0.125 <= recovery <= ppq * 4):
        raise ValueError('Dynamics values are outside their supported ranges.')
    return start, end, reaction, recovery


def _envelope_at(tick, hits, hit_ticks, length, parameters):
    start, end, reaction, recovery = parameters
    position = max(0.0, min(1.0, tick / length))
    baseline = start + (end - start) * position
    strongest = 0.0
    first = bisect.bisect_right(hit_ticks, tick - recovery)
    last = bisect.bisect_right(hit_ticks, tick)
    for hit_tick, velocity in hits[first:last]:
        age = tick - hit_tick
        dip = reaction * (velocity / 127.0) * (1.0 - age / recovery)
        strongest = max(strongest, dip)
    return max(1.0, min(127.0, baseline * (1.0 - strongest)))


def value_at(tick, drum_hits, ppq, section_length_ticks, dynamics):
    """Envelope value at a section-relative tick, in MIDI's 1..127 range.

    Simultaneous hits use the strongest currently recovering dip. Empty drum
    input produces only the requested gradual section envelope.
    """
    parameters = _parameters(dynamics, ppq, section_length_ticks)
    hits = sorted((float(when), max(0, min(127, int(velocity))))
                  for when, velocity in drum_hits if velocity > 0)
    return _envelope_at(tick, hits, [hit[0] for hit in hits],
                        section_length_ticks, parameters)


def shape_events(source_events, drum_hits, ppq, block_offset_ticks,
                 section_length_ticks, dynamics, channel, expression=True,
                 block_length_ticks=None):
    """Shape a repeated block and produce an independent velocity control lane.

    ``source_events`` are ``(block_tick, order, Message)`` records. Drum hits
    contain ``(section_tick, velocity)``; offsets keep envelopes continuous
    across repeated patterns. The optional explicit block length supports
    silent tails and pattern catalogs that omit end-of-track messages.

    CC11 expresses the same envelope as note velocity when enabled. Independent
    note-60 control pulses are returned separately for a velocity-to-VST mapper;
    they must not be mixed into the musical instrument track.
    """
    parameters = _parameters(dynamics, ppq, section_length_ticks)
    if type(channel) is not int or not 0 <= channel <= 15:
        raise ValueError('Dynamics require a valid MIDI channel.')
    events = list(source_events)
    if block_length_ticks is None:
        block_length_ticks = max((tick for tick, _, _ in events), default=0)
    if type(block_length_ticks) is not int or block_length_ticks <= 0:
        raise ValueError('Dynamics require a positive explicit block length.')
    if block_offset_ticks < 0 or block_offset_ticks + block_length_ticks > section_length_ticks:
        raise ValueError('Dynamics block must stay inside its section.')
    hits = sorted((float(when), max(0, min(127, int(velocity))))
                  for when, velocity in drum_hits if velocity > 0)
    hit_ticks = [hit[0] for hit in hits]

    def envelope(local_tick):
        return _envelope_at(block_offset_ticks + local_tick, hits, hit_ticks,
                            section_length_ticks, parameters)

    changed = 0
    processed = 0
    removed_cc11 = 0
    shaped = []
    for tick, order, message in events:
        if (expression and message.type == 'control_change'
                and message.channel == channel and message.control == 11):
            removed_cc11 += 1
            continue
        if message.type == 'note_on' and message.velocity and message.channel == channel:
            velocity = max(1, min(127, int(message.velocity * envelope(tick) / 127.0 + 0.5)))
            processed += 1
            changed += velocity != message.velocity
            message = message.copy(velocity=velocity)
        shaped.append((tick, order, message))

    # Sample grid stays aligned to the whole section, even for cropped blocks.
    grid = max(1, ppq / 4)
    first_index = math.ceil(block_offset_ticks / grid)
    last_index = math.ceil((block_offset_ticks + block_length_ticks) / grid)
    sample_ticks = {0}
    for index in range(first_index, last_index):
        local = int(index * grid + 0.5) - block_offset_ticks
        if 0 <= local < block_length_ticks:
            sample_ticks.add(local)
    recovery = parameters[3]
    for hit_tick, _ in hits:
        for when in (hit_tick, hit_tick + recovery):
            local = int(when + 0.5) - block_offset_ticks
            if 0 <= local < block_length_ticks:
                sample_ticks.add(local)
    samples = sorted(sample_ticks)
    controls = []
    sampled_values = []
    # Source initialization may contain CC121 (reset controllers). Establish
    # the planned expression after those same-tick messages so it cannot be
    # erased by an imported initialization sequence.
    expression_order = max((order for _, order, _ in events), default=0) + 1
    for index, tick in enumerate(samples):
        value = envelope(tick)
        sampled_values.append(value)
        midi_value = max(1, min(127, int(value + 0.5)))
        if expression:
            shaped.append((tick, expression_order + index,
                           mido.Message('control_change', channel=channel,
                                        control=11, value=midi_value)))
        # Map normalized 0..1 to valid nonzero note velocities, as expected by
        # the existing velocity-to-parameter protocol. The off is always first
        # at the next shared tick, and the final pulse ends exactly at the block.
        pulse_velocity = max(1, min(127, int(1 + 126 * value / 127.0 + 0.5)))
        end_tick = samples[index + 1] if index + 1 < len(samples) else block_length_ticks
        controls.append((tick, 1000000 + index * 2,
                         mido.Message('note_on', channel=channel, note=60,
                                      velocity=pulse_velocity)))
        controls.append((end_tick, -1000000 + index * 2,
                         mido.Message('note_off', channel=channel, note=60, velocity=0)))
    stats = {'notes_processed': processed, 'notes_changed': changed,
             'velocity_changes': changed,
             'cc11_replaced': removed_cc11,
             'cc11_generated': len(samples) if expression else 0,
             'automation_pulses': len(samples),
             'minimum_envelope': min(sampled_values),
             'maximum_envelope': max(sampled_values)}
    return (sorted(shaped, key=lambda item: (item[0], item[1])),
            sorted(controls, key=lambda item: (item[0], item[1])), stats)
