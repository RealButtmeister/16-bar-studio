"""Arrange an eight- or sixteen-bar performance using only its musical material.

Pitched tracks retain their source pitches and groove. Mapped Layer drums are
selected independently and receive small timing and velocity variations. No
extra notes or drum hits are added. The first peak preserves pitched tracks.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .model import (Arrangement, BAR, ControlChange, DRUM_ROLES, Note, PitchBend, PPQ, ROLES, Section,
                    Source, StudioError, Track)

PEAKS = frozenset(('hook', 'drop', 'chorus'))
LOW = frozenset(('bass', '808', 'sub'))
HARMONY = frozenset(('chords', 'pad'))
TOP = frozenset(('lead', 'pluck', 'arp', 'stab', 'screech', 'vocal'))
EFFECTS = frozenset(('fx', 'riser'))
GROOVE = DRUM_ROLES | LOW


def profiles() -> list[dict]:
    """Return fresh profile data; callers may safely add UI fields to it."""
    root = Path(__file__).resolve().parent.parent / 'profiles'
    return [json.loads((root / f'{name}.json').read_text(encoding='utf-8-sig'))
            for name in ('rap', 'pop', 'edm', 'hardstyle')]


def _roles(words: str) -> frozenset[str]:
    return frozenset(words.split())


# These recipes orchestrate the supplied performance. The legacy profiles'
# canned chords, beats, scales and automation are deliberately never generated.
# Intro texture, breakdown texture, supporting verse voice, shared kick/bass
# pickup rest (beats), section dynamics.
RECIPES = {
    'trap': ('chords pad pluck', 'pad chords pluck', 'pluck', .5, .87),
    'drill': ('pad pluck percussion', 'pad pluck fx', 'pluck', 1, .86),
    'boom-bap': ('kick snare closed_hat chords bass', 'chords bass percussion',
                 'lead', 0, .94),
    'melodic-trap': ('lead chords pad', 'lead pad chords vocal', 'lead', .5, .92),
    'dance-pop': ('chords pluck closed_hat', 'chords pad vocal', 'pluck', .5, .91),
    'synth-pop': ('pad arp closed_hat', 'pad arp lead', 'arp', .5, .90),
    'acoustic-pop': ('chords lead', 'chords lead vocal', 'lead', 0, .95),
    'rnb-pop': ('chords pad pluck', 'chords pad vocal', 'pluck', .5, .88),
    'progressive-house': ('kick closed_hat pluck', 'chords pad pluck lead',
                          'pluck', 1, .92),
    'tech-house': ('kick closed_hat percussion', 'stab vocal fx', 'stab', .5, .95),
    'deep-house': ('chords closed_hat bass', 'chords pad lead', 'lead', .5, .90),
    'uplifting-trance': ('kick bass arp closed_hat', 'lead pad chords arp',
                         'arp', 1, .92),
    'psy-trance': ('kick bass percussion', 'pad fx lead', 'arp', .5, .95),
    'euphoric': ('kick chords pluck', 'lead pad chords', 'lead', 1, .94),
    'raw': ('kick screech percussion', 'screech pad fx', 'screech', 2, .95),
    'classic': ('kick bass arp closed_hat', 'chords lead arp', 'arp', 1, .95),
}


def _lookup(genre: str, style: str) -> tuple[dict, str]:
    genre_key = str(genre).strip().casefold()
    style_key = str(style).strip().casefold()
    aliases = {'house / trance': 'edm', 'house/trance': 'edm', 'rap': 'rap',
               'hip-hop': 'rap', 'house': 'edm', 'trance': 'edm'}
    genre_key = aliases.get(genre_key, genre_key)
    for profile in profiles():
        if genre_key not in (profile['id'].casefold(), profile['name'].casefold()):
            continue
        for key, item in profile['styles'].items():
            if style_key in (key.casefold(), item['name'].casefold()):
                return profile, key
        raise StudioError(f'Unknown style for {profile["name"]}: {style}')
    raise StudioError(f'Unknown genre: {genre}')


def _sections(profile: dict, style: str, length: str, source_bars: int = 8) -> list[Section]:
    if length in ('source', 'eight'):
        kind = {'rap': 'hook', 'pop': 'chorus'}.get(profile['id'], 'drop')
        bars = source_bars if length == 'source' else 8
        return [Section(f'Original {bars} bars', kind, 0, bars, 1.0)]
    if length not in ('full', 'compact'):
        raise StudioError('Song length must be full, compact, or source.')
    result, counts, cursor = [], {}, 0
    for item in profile['arrangements'][length]:
        kind = item['kind']
        bars = 8 if kind == 'intro' else int(item['bars'])
        # A complete peak must expose the second half of a sixteen-bar upload,
        # including compact profiles whose original sections are all eight bars.
        if kind in PEAKS:
            bars = max(bars, source_bars)
        # The shorter tech-house breakdown is explicitly described by its profile.
        if kind == 'breakdown' and length == 'full':
            bars = int(profile['styles'][style].get('breakdown_bars', bars))
        counts[kind] = counts.get(kind, 0) + 1
        label = {'prechorus': 'Pre-chorus'}.get(kind, kind.title())
        result.append(Section(f'{label} {counts[kind]}', kind, cursor, bars,
                              float(profile['section_energy'].get(kind, .7))))
        cursor += bars
    return result


def _variation_level(value: str) -> int:
    key = str(value).strip().casefold()
    options = {'faithful': 0, 'minimal': 0, 'subtle': 0, 'balanced': 1,
               'bold': 2, 'adventurous': 2}
    if key not in options:
        raise StudioError('Variation must be Faithful, Balanced, or Bold.')
    return options[key]


def _choice(seed: int, *parts: object) -> int:
    # Stable across processes and Python versions; never uses global random state.
    data = '|'.join(map(str, (seed, *parts))).encode('utf-8')
    return int.from_bytes(hashlib.sha256(data).digest()[:4], 'big')


def _validate(source: Source, bpm: float) -> list[Track]:
    if type(source.bars) is not int or source.bars not in (8, 16):
        raise StudioError('This arranger needs one 8- or 16-bar source loop.')
    loop = source.bars * BAR
    if not math.isfinite(bpm) or not 20 <= bpm <= 400:
        raise StudioError('Tempo must be between 20 and 400 BPM.')
    tracks = [t for t in source.tracks if t.role != 'skip']
    if not tracks or not any(t.notes for t in tracks):
        raise StudioError('Assign at least one track with notes to a musical role.')
    ids = [t.id for t in tracks]
    if len(set(ids)) != len(ids):
        raise StudioError('Source track IDs must be unique.')
    for track in tracks:
        if track.role not in ROLES:
            raise StudioError(f'Choose a role for track "{track.name}" before generating.')
        if any(type(pitch) is not int or not 0 <= pitch <= 127 or role not in DRUM_ROLES
               for pitch, role in track.drum_map.items()):
            raise StudioError(f'Track "{track.name}" has an invalid Layer drum mapping.')
        for event in track.control_changes:
            if (any(type(v) is not int for v in (event.start, event.control, event.value, event.order))
                    or not 0 <= event.start <= loop
                    or not 0 <= event.control <= 127 or not 0 <= event.value <= 127):
                raise StudioError(f'Track "{track.name}" has an invalid MIDI controller event.')
        for bend in track.pitch_bends:
            if (not isinstance(bend.start, int) or isinstance(bend.start, bool)
                    or not isinstance(bend.value, int) or isinstance(bend.value, bool)
                    or type(bend.order) is not int
                    or not 0 <= bend.start <= loop or not -8192 <= bend.value <= 8191):
                raise StudioError(f'Track "{track.name}" has an invalid pitch-bend event.')
        for note in track.notes:
            if (not isinstance(note.start, int) or not isinstance(note.duration, int)
                    or note.start < 0 or note.start >= loop or note.duration <= 0
                    or type(note.start_order) is not int or type(note.end_order) is not int
                    or not 0 <= note.pitch <= 127 or not 1 <= note.velocity <= 127):
                raise StudioError(f'Track "{track.name}" has an invalid {source.bars}-bar note.')
    return tracks


def _active(role: str, bar: int, section: Section, recipe: tuple,
            family: str, style: str, present: set[str], level: int,
            occurrence: int, rotation: int) -> bool:
    intro, breakdown, support, _, _ = recipe
    kind, phase = section.kind, bar % 8
    late = bar >= section.bars // 2
    if kind == 'intro':
        if role in _roles(intro):
            return True
        if late and role in (GROOVE - {'cymbal', 'open_hat'}):
            return True
        return role in EFFECTS and late
    if kind in ('bridge', 'breakdown'):
        if role in _roles(breakdown):
            return True
        return late and role in ('percussion', 'closed_hat', 'riser')
    if kind == 'outro':
        if family in ('edm', 'hardstyle'):
            return role in GROOVE or (not late and role in HARMONY)
        return role in HARMONY or (not late and role in GROOVE | {support, 'vocal'})
    if kind in ('build', 'prechorus'):
        if role in HARMONY | {'snare', 'clap', 'closed_hat', 'riser', 'fx', support}:
            return True
        if role in LOW | {'kick'}:
            return not late or kind == 'prechorus'
        return late and role in TOP | {'open_hat', 'cymbal', 'percussion'}
    if kind == 'verse':
        if role in GROOVE | HARMONY | {'vocal'}:
            # Keep the whole performed groove, including its own swing and slides.
            return role not in ('open_hat', 'cymbal') or late
        if role in EFFECTS:
            return bar >= section.bars - 4
        if role == support:
            return True
        if role in TOP:
            if level == 0:
                return late
            span = 2 if level == 2 else 4
            answer = ((phase // span) + occurrence + rotation) % 2 == 1
            return answer if role in ('lead', 'screech') else not answer
        return False
    if kind in PEAKS:
        # Subsequent peaks develop in complete phrases, while the first is exact.
        if level and role in ('lead', 'pluck', 'screech'):
            partner = 'screech' if style == 'raw' else 'pluck'
            if 'lead' in present and partner in present and role in ('lead', partner):
                # Protect the arrival and the final phrase of a longer peak.
                if section.bars <= 8 or 8 <= bar < section.bars - 4:
                    span = 2 if level == 2 else 4
                    answer = ((phase // span) + rotation) % 2 == 1
                    return answer if role == partner else not answer
        return True
    return True


def _drum_active(role: str, pitch: int, local: int, section: Section,
                 recipe: tuple, family: str, style: str, present: set[str],
                 level: int, occurrence: int, rotation: int) -> bool:
    """Shape existing drum phrases, without probabilistic hit thinning or fills."""
    bar = local // BAR
    kind = section.kind
    late = bar >= section.bars // 2
    arrival = kind in PEAKS and bar < 2
    ending = bar >= section.bars - 2
    # Separate percussion rows can answer one another even when their labels
    # resolve to the same role; the mapping's actual pitches stay unchanged.
    phrase = (bar // 4 + occurrence + rotation + pitch) % 2
    active = _active(role, bar, section, recipe, family, style, present,
                     level, occurrence, rotation)
    if role == 'tom':
        # A transition can reveal only tom hits already written at this point.
        return (bar % 8 >= 6 or ending) and (kind not in ('intro', 'bridge', 'breakdown') or late)
    if role == 'shaker':
        if kind in ('bridge', 'breakdown', 'outro'):
            return late and phrase == 1 and kind != 'outro'
        if kind == 'intro':
            return late and phrase == 1
        return (arrival or late or phrase == 1) and (level == 0 or phrase == 1 or arrival or ending)
    if role in ('open_hat', 'cymbal'):
        if arrival:
            return True
        if kind == 'intro':
            return ending
        if kind in PEAKS:
            return ending or bar % 4 >= (2 if role == 'open_hat' else 3)
        return active and (late if role == 'open_hat' else ending)
    if role == 'clap':
        if kind == 'intro':
            return active and ending
        if kind in ('verse', 'bridge', 'breakdown'):
            return active and (late if level == 0 else phrase == 1)
        if kind in PEAKS:
            return arrival or ending or level == 0 or phrase == 1
    if role == 'percussion':
        return active and (arrival or ending or level == 0 or phrase == 1)
    if role == 'closed_hat' and active:
        # Leave occasional breathing space at phrase endings; never insert a
        # roll or redistribute deleted hits to a different rhythmic position.
        if level and not arrival and kind in ('verse', 'build', 'prechorus'):
            return not (bar % 8 == 7 and local % BAR >= BAR - PPQ)
    return active


def _humanize_layer(track: Track, source: Source, sections: list[Section],
                     seed: int, level: int, tempo: float) -> bool:
    """Move existing hits a few milliseconds, preserving pitch and event count.

    Downbeat boundaries and midpoint bounds keep hit order intact. Kick, snare
    and clap share their timing choice at each source onset, so layered accents
    stay together. Durations are only shortened to avoid a same-pitch overlap
    or the final boundary. The return value reports duration shortening.
    """
    if not track.notes:
        return False
    notes = sorted(track.notes, key=lambda n: (n.start, n.start_order, n.pitch))
    by_pitch: dict[int, list[int]] = {}
    for index, note in enumerate(notes):
        by_pitch.setdefault(note.pitch, []).append(index)
    bounds = {}
    pitch_groups = []
    for indices in by_pitch.values():
        # Treat source duplicates at the same onset as a group. They are never
        # multiplied or silently converted into a newly staggered flam.
        groups = []
        for index in indices:
            if not groups or notes[groups[-1][0]].start != notes[index].start:
                groups.append([])
            groups[-1].append(index)
        pitch_groups.append(groups)
        for group_index, group in enumerate(groups):
            start = notes[group[0]].start
            low, high = start // BAR * BAR, (start // BAR + 1) * BAR - 1
            if group_index:
                low = max(low, (notes[groups[group_index - 1][0]].start + start) // 2 + 1)
            if group_index + 1 < len(groups):
                high = min(high, (start + notes[groups[group_index + 1][0]].start) // 2)
            for index in group:
                bounds[index] = (low, high)
    # Intersect the safe range for simultaneous backbone accents. A dense
    # snare retrigger must constrain its layered kick/clap by the same amount.
    core_bounds = {}
    for index, note in enumerate(notes):
        if track.drum_map.get(note.pitch) in ('kick', 'snare', 'clap'):
            low, high = bounds[index]
            if note.start in core_bounds:
                previous_low, previous_high = core_bounds[note.start]
                low, high = max(low, previous_low), min(high, previous_high)
            core_bounds[note.start] = (low, high)
    song_end = sections[-1].end_bar * BAR
    result = []
    section_index = 0
    for index, note in enumerate(notes):
        while note.start >= sections[section_index].end_bar * BAR:
            section_index += 1
        section = sections[section_index]
        local = note.start - section.start_bar * BAR
        cycle, onset = divmod(local, source.bars * BAR)
        role = track.drum_map.get(note.pitch, 'percussion')
        core = role in ('kick', 'snare', 'clap')
        key = (source.sha256 or source.path, track.id, section_index, cycle, onset)
        timing_key = 'backbone' if core else note.pitch
        milliseconds = (3 if core else {'closed_hat': 8, 'shaker': 10,
                         'open_hat': 7, 'tom': 5}.get(role, 6))
        span = max(1, round(milliseconds * PPQ * tempo / 60000 * (.8 + .2 * level)))
        delta = _choice(seed, *key, timing_key, 'timing') % (2 * span + 1) - span
        low, high = core_bounds[note.start] if core else bounds[index]
        start = max(low, min(high, note.start + delta))
        # Distinct drum sounds get their own soft velocity movement even when
        # a layered kick/snare/clap accent shares exactly the same timing.
        velocity_span = (3 if core else 5) + level
        velocity_delta = (_choice(seed, *key, note.pitch, note.start_order, 'velocity')
                          % (2 * velocity_span + 1) - velocity_span)
        velocity = max(1, min(127, note.velocity + velocity_delta))
        result.append(Note(start, min(note.duration, song_end - start), note.pitch, velocity,
                           note.start_order, note.end_order))
    clipped = any(before.duration != after.duration for before, after in zip(notes, result))
    for groups in pitch_groups:
        for position, group in enumerate(groups):
            following = groups[position + 1] if position + 1 < len(groups) else []
            next_start = result[following[0]].start if following else song_end
            for index in group:
                note = result[index]
                duration = min(note.duration, next_start - note.start)
                end_order = note.end_order
                if following and note.start + duration == next_start:
                    end_order = min(end_order, min(result[i].start_order for i in following) - 1)
                if duration != note.duration or end_order != note.end_order:
                    clipped |= duration != note.duration
                    result[index] = Note(note.start, duration, note.pitch, note.velocity,
                                         note.start_order, end_order)
    track.notes = result
    return clipped


def _copy(track: Track) -> Track:
    return Track(track.id, track.name, track.role, track.channel, track.program,
                 [], track.instrument_name, list(track.controls), program_order=track.program_order,
                 drum_map=dict(track.drum_map), drum_names=dict(track.drum_names))


def _performance_cycles(tracks, sections, source_bars=8):
    orders = [0]
    for track in tracks:
        orders.append(track.program_order)
        orders.extend(event.order for event in track.pitch_bends)
        orders.extend(event.order for event in track.control_changes)
        for note in track.notes:
            orders.extend((note.start_order, note.end_order))
    largest = max(abs(value) for value in orders)
    stride, shift = 2 * largest + 4096, largest + 1024
    cycles = []
    for section in sections:
        for loop_bar in range(0, section.bars, source_bars):
            cycles.append(((section.start_bar + loop_bar) * BAR,
                           min(source_bars * BAR, (section.bars - loop_bar) * BAR),
                           len(cycles) * stride))
    return cycles, shift, len(cycles) * stride + shift


def _repeat_pitch_bends(track: Track, cycles, shift, final_order) -> list[PitchBend]:
    """Keep channel expression on the same source clock as each repeated phrase.

    Bends also run through note rests: they are channel state, and filtering
    them by note onset would change sustained notes and later entrances.
    A new cycle starts in the source's initial state, never the prior cycle's
    final bend. MIDI's default initial pitch-wheel value is centered (zero).
    """
    if not track.pitch_bends:
        return []
    source = sorted(track.pitch_bends, key=lambda bend: (bend.start, bend.order))
    has_initial = source[0].start == 0
    result = []
    for start, available, base in cycles:
        if not has_initial:
            result.append(PitchBend(start, 0, base + 900))
        for bend in source:
            if bend.start > available:
                break
            result.append(PitchBend(start + bend.start, bend.value, base + shift + bend.order))
    # Leave the receiving instrument in tune after the song's final note-off.
    if result and result[-1].value != 0:
        result.append(PitchBend(cycles[-1][0] + cycles[-1][1], 0, final_order + 10))
    return result


def _repeat_controllers(track: Track, cycles, shift, final_order) -> list[ControlChange]:
    if not track.control_changes:
        return []
    source = sorted(track.control_changes, key=lambda event: (event.start, event.order))
    used = {event.control for event in source}
    initial = {event.control for event in source if event.start == 0}
    # These are conventional MIDI/GM starting values. Unknown/device-specific
    # controls keep their explicit source stream instead of invented defaults.
    defaults = {1: 0, 7: 100, 10: 64, 11: 127, 33: 0, 39: 0, 42: 0, 43: 0,
                64: 0, 65: 0, 66: 0, 67: 0, 68: 0, 69: 0}
    defaults.update(dict(track.controls))
    pedals = sorted(used & {64, 66, 69})
    result = []
    for start, available, base in cycles:
        for index, control in enumerate(pedals):
            result.append(ControlChange(start, control, 0, base + 100 + index))
        for index, control in enumerate(sorted(used - initial - set(pedals))):
            if control in defaults and control not in {6, 38, 96, 97, 98, 99, 100, 101}:
                result.append(ControlChange(start, control, defaults[control], base + 200 + index))
        # A data-entry command without any initial selection must not inherit
        # the preceding cycle's RPN/NRPN selection. Explicit setup stays intact.
        if used & {6, 38, 96, 97, 98, 99, 100, 101} and not initial & {98, 99, 100, 101}:
            for index, control in enumerate((99, 98, 101, 100)):
                result.append(ControlChange(start, control, 127, base + 400 + index))
        for event in source:
            if event.start > available:
                break
            result.append(ControlChange(start + event.start, event.control, event.value,
                                        base + shift + event.order))
    end = cycles[-1][0] + cycles[-1][1]
    for index, control in enumerate(pedals):
        result.append(ControlChange(end, control, 0, final_order + 20 + index))
    return sorted(result, key=lambda event: (event.start, event.order))


def make_arrangement(source: Source, genre: str, style: str,
                     length: str = 'full', variation: str = 'Balanced',
                     seed: int = 2026, bpm: float | None = None) -> Arrangement:
    """Build a deterministic song from synchronized source phrases.

    Note selections are based on onset; sustained notes are allowed to ring
    across phrase/section boundaries. Only the final song boundary clips tails.
    Tempo defaults to the uploaded MIDI's tempo rather than a genre preset.
    """
    profile, style = _lookup(genre, style)
    level = _variation_level(variation)
    tempo = source.bpm if bpm is None else float(bpm)
    tracks = _validate(source, tempo)
    sections = _sections(profile, style, length, source.bars)
    cycles, order_shift, final_order = _performance_cycles(tracks, sections, source.bars)
    cycle_bases = {start: base for start, _, base in cycles}
    recipe = RECIPES[style]
    output = [_copy(track) for track in tracks]
    warnings = list(source.warnings)
    decisions = []
    song_end = sections[-1].end_bar * BAR
    track_roles = {track.role for track in tracks if track.notes}
    present = {role for track in tracks if track.notes
               for role in (set(track.drum_map.values()) if track.drum_map else {track.role})}
    priority = ('chords', 'pad', 'lead', 'pluck', 'vocal', 'arp', 'bass', '808',
                'sub', 'kick', 'snare', 'percussion', 'closed_hat')
    core = min((t for t in tracks if t.notes),
               key=lambda t: priority.index(t.role) if t.role in priority else 99)
    peak_seen, occurrences = False, {}
    clipped = False
    for section_index, section in enumerate(sections):
        kind = section.kind
        occurrences[kind] = occurrences.get(kind, 0) + 1
        exact = kind in PEAKS and not peak_seen
        if kind in PEAKS:
            peak_seen = True
        rotation = _choice(seed, style, section_index) % 2
        section_notes: dict[str, list[Note]] = {t.id: [] for t in tracks}
        gap = (int(recipe[3] * PPQ) if level and not exact
               and kind in ('build', 'prechorus', 'verse') else 0)
        # A transition removes only new low-end onsets, never cuts a bass tail.
        # A single source track remains a complete musical backbone.
        if len([t for t in tracks if t.notes]) == 1:
            gap = 0
        for track in tracks:
            for loop_bar in range(0, section.bars, source.bars):
                order_base = cycle_bases[(section.start_bar + loop_bar) * BAR] + order_shift
                for original in track.notes:
                    local = loop_bar * BAR + original.start
                    if local >= section.bars * BAR:
                        continue
                    role = track.drum_map.get(original.pitch, 'percussion') if track.drum_map else track.role
                    if track.drum_map:
                        selected = _drum_active(
                            role, original.pitch, local, section, recipe, profile['id'],
                            style, present, level, occurrences[kind], rotation)
                    else:
                        selected = exact or _active(
                            role, local // BAR, section, recipe, profile['id'],
                            style, track_roles, level, occurrences[kind], rotation)
                    if not track.drum_map and track.id == core.id and len(track_roles) <= 2:
                        selected = True
                    if gap and role in LOW | {'kick'}:
                        selected = selected and local < section.bars * BAR - gap
                    if not selected:
                        continue
                    absolute = section.start_bar * BAR + local
                    duration = min(original.duration, song_end - absolute)
                    clipped |= duration != original.duration
                    if exact and not track.drum_map:
                        velocity = original.velocity
                    else:
                        factor = recipe[4] * (.82 + .18 * section.energy)
                        if kind in ('build', 'prechorus'):
                            factor += .05 * local / (section.bars * BAR)
                        velocity = max(1, min(127, round(original.velocity * factor)))
                    section_notes[track.id].append(
                        Note(absolute, duration, original.pitch, velocity,
                             order_base + original.start_order, order_base + original.end_order))
        # If a sparse upload lacks this recipe's textures, retain a complete
        # source voice; do not invent a pad, drum or chord progression.
        fallback = False
        if not any(section_notes.values()):
            fallback = True
            fallback_notes = core.notes
            if core.drum_map:
                # A drum-only upload can lack the recipe's opening texture.
                # Keep one performed drum row as its pulse, never restore the
                # whole combined Layer merely to avoid an empty section.
                candidates = [note for note in core.notes if note.start < section.bars * BAR]
                preferred = ('kick', 'closed_hat', 'snare', 'shaker', 'percussion',
                             'clap', 'open_hat', 'tom', 'cymbal')
                if candidates:
                    pitch = min({note.pitch for note in candidates}, key=lambda value: (
                        preferred.index(core.drum_map.get(value, 'percussion')), value))
                    fallback_notes = [note for note in candidates if note.pitch == pitch]
                else:
                    fallback_notes = []
            for loop_bar in range(0, section.bars, source.bars):
                order_base = cycle_bases[(section.start_bar + loop_bar) * BAR] + order_shift
                for note in fallback_notes:
                    local = loop_bar * BAR + note.start
                    if local < section.bars * BAR:
                        start = section.start_bar * BAR + local
                        duration = min(note.duration, song_end - start)
                        clipped |= duration != note.duration
                        section_notes[core.id].append(
                            Note(start, duration, note.pitch, note.velocity,
                                 order_base + note.start_order, order_base + note.end_order))
        for track in output:
            track.notes.extend(section_notes[track.id])
        decisions.append({
            'section': section.name, 'kind': kind, 'start_bar': section.start_bar,
            'bars': section.bars, 'source_cycle_bars': source.bars,
            'exact_source': exact and not any(t.drum_map and t.notes for t in tracks),
            'pitched_source_preserved': exact,
            'active_tracks': [t.id for t in tracks if section_notes[t.id]],
            'phrase_activity': {
                t.id: [b for b in range(0, section.bars, 4)
                       if any((section.start_bar + b) * BAR <= n.start
                              < (section.start_bar + min(b + 4, section.bars)) * BAR
                              for n in section_notes[t.id])]
                for t in tracks if section_notes[t.id]},
            'low_end_pickup_rest_beats': gap / PPQ,
            'added_musical_notes': 0, 'fallback_source_voice': fallback,
            'drum_activity': {
                t.id: {str(pitch): {
                    'name': t.drum_names.get(pitch, role), 'role': role,
                    'hits': sum(note.pitch == pitch for note in section_notes[t.id])}
                    for pitch, role in t.drum_map.items()}
                for t in tracks if t.drum_map},
            'drum_humanization': any(t.drum_map and section_notes[t.id] for t in tracks),
            'complete_source_peak': kind in PEAKS and section.bars >= source.bars,
            'approach': ('Complete original pitched performance; independent existing Layer drums'
                         if exact and any(t.drum_map for t in tracks) else
                         'Complete original performance and velocity' if exact else
                         f'Shared {source.bars}-bar loop; phrase entrances, rests and dynamics'),
        })
    source_by_id = {track.id: track for track in tracks}
    drum_clipped = False
    for track in output:
        if track.drum_map:
            drum_clipped |= _humanize_layer(track, source, sections, seed, level, tempo)
        track.notes.sort(key=lambda note: (note.start, note.start_order, note.pitch, note.duration, note.velocity))
        before = source_by_id[track.id]
        track.pitch_bends = _repeat_pitch_bends(before, cycles, order_shift, final_order)
        track.control_changes = _repeat_controllers(before, cycles, order_shift, final_order)
        track.program_order = order_shift + before.program_order
    if clipped:
        warnings.append('Notes extending beyond the final song boundary were shortened there; '
                        'internal phrase and section tails were retained.')
    if drum_clipped:
        warnings.append('Layer drum note lengths were shortened where needed to avoid repeated-pitch '
                        'overlaps or stay within the final song boundary after humanization.')
    return Arrangement(output, sections, tempo, profile['id'], style, int(seed),
                       warnings, decisions)
