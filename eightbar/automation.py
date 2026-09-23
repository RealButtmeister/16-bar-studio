"""Section-held pitches and independent, straight-segment velocity envelopes."""
from bisect import bisect_right
from dataclasses import dataclass
import math
from .model import Arrangement, Note, StudioError, PPQ, BAR

DENSITIES = (2, 4, 8, 16, 120, 240, 480, 960)
AUTOMATION_STYLES = ('Clean lines', 'Classic')
SHAPES = ('Rise', 'Hold', 'Fall', 'Rise and hold', 'Hold and fall', 'Triangle',
          'Step up', 'Step down', 'Intro rise', 'Hold and lift')
PROTOCOL = {
    'id': 'eightbar.velocity.v1', 'version': 1,
    'control_message': 'note_on', 'velocity_range': [1, 127],
    'normalized_value': '(midi_velocity - 1) / 126',
    'vst3_normalized_note_velocity_to_value': 'clamp((event.velocity * 127 - 1) / 126, 0, 1)',
    'note_pitch': 'constant per section; independent tag, never the control amount',
    'note_off': 'hold last control value; do not reset or release the parameter',
    'velocity_zero': 'note-off; ignore for control changes',
    'interpolation': 'sender samples piecewise linear segments; receiver uses sample-and-hold',
    'smoothing_default': 'off; optional plugin smoothing must not alter the stored MIDI',
    'routing': 'one automation MIDI per instrument; route to its own VST3 instance/input',
    'runtime_track_names': 'labels for DAW import only; receiver does not depend on file metadata',
    'transport_stop': 'hold last value; explicit plugin Reset sets neutral value',
    'initial_value': 1.0, 'neutral_value': 1.0,
    'musical_velocity_shaping': 'never applied to automation',
}

@dataclass
class Envelope:
    section_index: int
    pitch: int
    points: list[tuple[float, float]]
    shape: str = 'Custom'

def preset_points(shape, low=0.0, high=1.0):
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= 1 and 0 <= high <= 1):
        raise StudioError('Velocity levels must be between 0 and 100%.')
    mid = low + (high-low)*0.32
    shapes = {
        'Intro rise': [(0,low),(1,high)], 'Rise': [(0,low),(1,high)],
        'Fall': [(0,high),(1,low)], 'Hold': [(0,high),(1,high)],
        'Triangle': [(0,low),(.5,high),(1,low)],
        'Rise and hold': [(0,low),(.75,high),(1,high)],
        'Hold and fall': [(0,high),(.25,high),(1,low)],
        'Step up': [(0,low),(.5,low),(.5,high),(1,high)],
        'Step down': [(0,high),(.5,high),(.5,low),(1,low)],
        'Hold and lift': [(0,low),(.42,low),(.5,mid),(.76,mid),(.86,low+(high-low)*.43),
                          (.93,low+(high-low)*.65),(1,high)],
    }
    if shape not in shapes:
        raise StudioError('Unknown automation shape: '+str(shape))
    return shapes[shape]

def default_envelopes(arrangement, track=None, style='Classic'):
    """Classic remains available for older callers; the app selects Clean lines.

    These are control envelopes for Velocity Pass, separate from the music.
    The musical arrangement supplies the rests. Control keeps following song
    time through rests, so reentries receive the right value and FX can decay.
    """
    if style == 'Clean lines':
        return clean_envelopes(arrangement, track)
    if style != 'Classic':
        raise StudioError('Unknown automation style: '+str(style))
    result=[]
    for i,section in enumerate(arrangement.sections):
        if i==0 and section.kind=='intro': shape,lo,hi,pitch='Intro rise',0.,1.,60
        elif section.kind in ('verse','breakdown','bridge','break'):
            shape,lo,hi,pitch='Hold and lift',.5,.95,57
        elif section.kind in ('build','prechorus'):
            shape,lo,hi,pitch='Rise',.45,1.,60
        elif section.kind=='outro': shape,lo,hi,pitch='Fall',0.,.85,57
        else: shape,lo,hi,pitch='Hold',1.,1.,60
        # Section pitches are independent tags for the future receiver. A
        # visible step marks each new section; every pitch is user-editable.
        pitch=60 if i%2==0 else 57
        result.append(Envelope(i,pitch,preset_points(shape,lo,hi),shape))
    return result


def clean_envelopes(arrangement, track=None):
    """A few straight ramps and plateaus, scaled to the actual song sections.

    Pads swell more deeply; melodic parts rise through the opening; the rhythm
    section stays steadier. Only lower-energy sections reset the level. Adjacent
    build sections continue the preceding rise instead of restarting it.
    """
    from .model import DRUM_ROLES
    role = getattr(track, 'role', 'pad')
    drums = role in DRUM_ROLES or bool(getattr(track, 'drum_map', {}))
    bass = role in ('bass', '808', 'sub')
    pad = role in ('pad', 'fx', 'riser')
    steady = drums or bass or role == 'vocal'
    result = []
    previous = 1.0
    verses = 0
    sections = arrangement.sections
    for i, section in enumerate(sections):
        kind = section.kind
        next_kind = sections[i + 1].kind if i + 1 < len(sections) else ''
        if kind == 'intro':
            if pad:
                low, high = 0., 1.
            elif steady:
                low, high = (.65, .90) if drums else (.45, .80)
            else:
                low, high = .18, .48
            if i and sections[i - 1].kind == 'intro':
                low = previous
            shape = 'Rise'
        elif kind == 'verse':
            verses += 1
            if steady:
                low = .80 if drums else .70
            elif pad:
                low = .50 if verses == 1 else .30
            else:
                low = previous if i and sections[i - 1].kind == 'intro' else .35
            high = .82 if next_kind in ('build', 'prechorus') else 1.
            shape = 'Rise'
        elif kind in ('bridge', 'breakdown', 'break'):
            low = .70 if steady else (.25 if pad else .32)
            high = .82 if next_kind in ('build', 'prechorus') else 1.
            shape = 'Rise'
        elif kind in ('build', 'prechorus'):
            low = previous if result else (.70 if steady else .35)
            high = 1.
            shape = 'Rise'
        elif kind == 'outro':
            low = (.60 if drums else .35) if steady else (0. if pad else .20)
            high = previous
            shape = 'Fall'
        else:
            low = high = 1.
            shape = 'Hold'
        points = preset_points(shape, low, high)
        previous = points[-1][1]
        result.append(Envelope(i, 60 if i % 2 == 0 else 57, points, shape))
    return result

def validate_envelopes(arrangement,envelopes):
    if len(envelopes)!=len(arrangement.sections):
        raise StudioError('Each song section needs one automation envelope.')
    for i,env in enumerate(envelopes):
        if env.section_index!=i or type(env.pitch) is not int or not 0<=env.pitch<=127:
            raise StudioError('Section automation needs a MIDI note from 0 to 127.')
        if not 2<=len(env.points)<=256:
            raise StudioError('An envelope needs 2 to 256 points.')
        previous=-1
        for x,y in env.points:
            if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in (x,y)):
                raise StudioError('Automation points must be finite numbers.')
            if not previous<=x<=1 or not 0<=y<=1 or x<0:
                raise StudioError('Automation points must be ordered and between 0 and 100%.')
            previous=x
        if env.points[0][0]!=0 or env.points[-1][0]!=1:
            raise StudioError('An envelope must begin at 0% time and end at 100% time.')

def velocity_at(points,position):
    xs=[p[0] for p in points]
    i=bisect_right(xs,position)-1
    if i<0:return points[0][1]
    if i>=len(points)-1:return points[-1][1]
    x0,y0=points[i];x1,y1=points[i+1]
    return y0+(y1-y0)*(position-x0)/(x1-x0)

def midi_velocity(value):
    return max(1,min(127,int(math.floor(1+126*value+.5))))

def iter_notes(arrangement,envelopes,density=2):
    validate_envelopes(arrangement,envelopes)
    if type(density) is not int or density not in DENSITIES:
        raise StudioError('Choose one of the available automation note densities.')
    step=PPQ//density
    for section,env in zip(arrangement.sections,envelopes):
        begin=section.start_bar*BAR; length=section.bars*BAR
        # Use the first and last note-on as envelope endpoints, so the final
        # visible note reaches the exact target before the next section.
        span=max(1,length-step)
        # Off-grid corners split a note rather than moving the intended corner.
        quantized=[(round(x*span),y) for x,y in env.points]
        boundaries=set(range(0,length,step))
        boundaries.update(x for x,_ in quantized)
        ordered=sorted(boundaries)
        for j,offset in enumerate(ordered):
            end=ordered[j+1] if j+1<len(ordered) else length
            if end>offset:
                value=velocity_at(quantized,offset)
                yield Note(begin+offset,end-offset,env.pitch,midi_velocity(value))

def note_name(pitch):
    return ('C','C#','D','D#','E','F','F#','G','G#','A','A#','B')[pitch%12]+str(pitch//12)

def parse_note(text):
    import re
    value=str(text).strip()
    if value.isdigit():pitch=int(value)
    else:
        match=re.fullmatch(r'([A-Ga-g])([#b]?)(\d{1,2})',value)
        if not match:raise StudioError('Use an FL note name such as C5 or A4, or MIDI number 0–127.')
        letter,acc,octave=match.groups()
        pitch=int(octave)*12+{'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[letter.upper()]+{'':0,'#':1,'b':-1}[acc]
    if not 0<=pitch<=127:raise StudioError('The section note must be MIDI 0–127.')
    return pitch
