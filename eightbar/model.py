from dataclasses import dataclass, field

PPQ = 960
BAR = 4 * PPQ
ROLES = ('kick', 'snare', 'clap', 'closed_hat', 'open_hat', 'shaker', 'tom', 'cymbal',
         'percussion', 'bass', '808', 'sub', 'lead', 'chords', 'pad',
         'arp', 'pluck', 'stab', 'screech', 'fx', 'riser', 'vocal', 'skip')
DRUM_ROLES = frozenset(('kick', 'snare', 'clap', 'closed_hat', 'open_hat',
                        'shaker', 'tom', 'cymbal', 'percussion'))

class StudioError(ValueError):
    pass

@dataclass(frozen=True)
class Note:
    start: int
    duration: int
    pitch: int
    velocity: int
    start_order: int = field(default=0, compare=False)
    end_order: int = field(default=0, compare=False)

    @property
    def end(self):
        return self.start + self.duration

@dataclass(frozen=True)
class PitchBend:
    start: int
    value: int  # signed MIDI pitch wheel, -8192..8191; 0 is centered
    order: int = field(default=0, compare=False)

@dataclass(frozen=True)
class ControlChange:
    start: int
    control: int
    value: int
    order: int = 0

@dataclass
class Track:
    id: str
    name: str
    role: str
    channel: int = 0  # zero-based MIDI channel, 0..15
    program: int | None = None
    notes: list[Note] = field(default_factory=list)
    instrument_name: str = ''
    controls: list[tuple[int, int]] = field(default_factory=list)
    pitch_bends: list[PitchBend] = field(default_factory=list)
    control_changes: list[ControlChange] = field(default_factory=list)
    program_order: int = 0
    drum_map: dict[int, str] = field(default_factory=dict)
    drum_names: dict[int, str] = field(default_factory=dict)

@dataclass
class Source:
    path: str
    tracks: list[Track]
    bpm: float = 130.0
    bars: int = 8
    warnings: list[str] = field(default_factory=list)
    sha256: str = ''

@dataclass(frozen=True)
class Section:
    name: str
    kind: str
    start_bar: int
    bars: int
    energy: float

    @property
    def end_bar(self):
        return self.start_bar + self.bars

@dataclass
class Arrangement:
    tracks: list[Track]
    sections: list[Section]
    bpm: float
    genre: str
    style: str
    seed: int
    warnings: list[str] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)

    @property
    def bars(self):
        return self.sections[-1].end_bar if self.sections else 0
