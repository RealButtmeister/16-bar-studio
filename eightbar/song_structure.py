"""Portable, versioned song map consumed first by Pop Indie Writer."""
from __future__ import annotations

import math
import re
import unicodedata


FORMAT = '16bar.song_structure'
VERSION = 1


def _label(value, limit=160):
    value = ''.join(' ' if unicodedata.category(char) in ('Cc', 'Cf', 'Cs') else char for char in value)
    value = re.sub(r'[\[\]\r\n\x00-\x1f]', ' ', value)
    return ' '.join(value.split())[:limit].strip()


def build_song_structure(plan, documents):
    """Translate validated MIDI sections without guessing bars from lyric lines.

    No filenames, credentials, raw MIDI, or software-specific note events are
    needed by a lyric writer. Each section carries its own tempo so timestamps
    remain correct when A and B use different tempos.
    """
    sections = []
    start_bar = 1
    start_seconds = 0.0
    for index, section in enumerate(plan['sections']):
        document = documents[section['pattern'][0]]
        bars = document.bars * section['repeats']
        duration = bars * 240 / document.bpm
        instrumental = section['kind'] in ('intro', 'outro')
        instruments = [document.midi.tracks[track].name or f'Track {track + 1}'
                       for track in section['active_tracks']
                       if any(note.track == track for note in document.notes)]
        sections.append({
            'id': f'section_{index + 1:02d}',
            'name': (_label(section['name'], 80) if any(char.isalnum() for char in _label(section['name'], 80))
                     else f'Section {index + 1}'),
            'kind': section['kind'], 'start_bar': start_bar,
            'end_bar': start_bar + bars - 1, 'bars': bars,
            'start_seconds': round(start_seconds, 6),
            'end_seconds': round(start_seconds + duration, 6),
            'bpm': document.bpm, 'pattern': section['pattern'],
            'vocal_mode': 'instrumental' if instrumental else 'lyrics',
            'suggested_lines': 0 if instrumental else max(1, min(64, math.ceil(bars / 2))),
            'notes': _label(section['reason'], 2000),
            'instruments': [_label(name) or f'Instrument {i + 1}' for i, name in enumerate(instruments)],
        })
        start_bar += bars
        start_seconds += duration
    # A phrase suggestion is editable, not a statement that two bars must equal
    # one line. Keep even long supported arrangements within the lyric budget.
    while sum(section['suggested_lines'] for section in sections) > 160:
        largest = max(sections, key=lambda section: section['suggested_lines'])
        largest['suggested_lines'] -= 1
    return {'format': FORMAT, 'version': VERSION,
            'title': _label(plan['title']) or 'Untitled song', 'genre': 'pop',
            'time_signature': [4, 4], 'total_bars': plan['total_bars'],
            'duration_seconds': round(start_seconds, 6), 'sections': sections}
