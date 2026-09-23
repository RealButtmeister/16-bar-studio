"""velocity_shaper.py — bake role-aware velocity dynamics into any MIDI file.

Usage:  python velocity_shaper.py <song.mid> [out.mid]

Reads instrument/track names, classifies each track's role, and rewrites ONLY
note-on velocities (timing/pitch/CC untouched). Output defaults to
<song>_VEL.mid next to the source.

Per-role treatment:
  kick        hard & consistent, downbeat accents
  snare/clap  backbeat accents; fast rolls become crescendos
  hats/shaker offbeat accents, low base, humanize
  perc/drums  beat-position accents, humanize
  bass        solid, beat-1 accents, follows the arc
  lead/melody phrase-shaped: rise toward phrase peaks, accent high & long notes
  pad/string  slow swell across each note/section
  pluck/arp   alternating accents, light humanize
  guitar      downbeat/downstroke accents, chord stabs even
  fx/riser    crescendo across the note run
Unknown tracks are classified by pitch range / density / polyphony.

A global energy arc (per-bar note density across all tracks, smoothed) scales
everything, so quiet sections sit lower and busy sections push toward max.
"""
import sys, os, math, random
import mido

random.seed(7)

if len(sys.argv) < 2:
    print(__doc__); sys.exit(1)
SRC = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(SRC)[0] + "_VEL.mid"

m = mido.MidiFile(SRC)
PPQ = m.ticks_per_beat
BAR = PPQ * 4

# ---------------- gather notes per track ----------------
class Note:
    __slots__ = ('t','pitch','vel','dur','msg','vel_new')
    def __init__(s, t, pitch, vel, msg): s.t=t; s.pitch=pitch; s.vel=vel; s.dur=0; s.msg=msg

track_info = []
song_end = 1
for tr in m.tracks:
    t = 0; name = None; notes = []; open_n = {}
    for msg in tr:
        t += msg.time
        if msg.type == 'track_name': name = msg.name
        elif msg.type == 'note_on' and msg.velocity > 0:
            n = Note(t, msg.note, msg.velocity, msg); notes.append(n)
            open_n.setdefault(msg.note, []).append(n)
            song_end = max(song_end, t+1)
        elif msg.type in ('note_off','note_on'):
            lst = open_n.get(msg.note)
            if lst: lst.pop(0).dur = t  # temp: absolute off time
    for n in notes:
        n.dur = max(1, (n.dur - n.t) if n.dur > n.t else PPQ//4)
    track_info.append({'name': name or '', 'notes': notes})

n_bars = math.ceil(song_end / BAR)

# ---------------- role classification ----------------
def classify(name, notes):
    n = name.lower()
    for keys, role in [
        (('kick','bd','bassdrum'), 'kick'), (('snare','clap','rim'), 'snare'),
        (('hat','hh','shaker','tamb','cymbal','ride','crash'), 'hat'),
        (('drum','perc','tom','layer','beat'), 'perc'),
        (('bass','sub','808'), 'bass'),
        (('pad','string','strings','choir','ens','swell','atmo'), 'pad'),
        (('pluck','arp'), 'pluck'),
        (('guitar','gtr'), 'guitar'),
        (('riser','uplift','sweep','fx','noise','impact'), 'fx'),
        (('lead','melody','vocal','flute','horn','brass','piano','key','synth','chord'), 'lead'),
    ]:
        if any(k in n for k in keys): return role
    if not notes: return 'lead'
    pitches = [x.pitch for x in notes]
    lo, hi = min(pitches), max(pitches)
    dens = len(notes) / max(n_bars, 1)
    avg_dur = sum(x.dur for x in notes) / len(notes)
    if hi < 48 and dens > 1: return 'bass'
    if hi - lo <= 5 and dens >= 8: return 'perc'      # repeated few pitches, busy
    if avg_dur > PPQ*2: return 'pad'
    if dens >= 12: return 'pluck'
    return 'lead'

# ---------------- global energy arc ----------------
dens_bar = [0]*n_bars
for ti in track_info:
    for x in ti['notes']: dens_bar[min(x.t // BAR, n_bars-1)] += 1
mx = max(max(dens_bar), 1)
arc = [d/mx for d in dens_bar]
# smooth (2-bar window) and lift floor so quiet sections aren't inaudible
def smooth(a, w):
    out=[]
    for i in range(len(a)):
        lo,hi = max(0,i-w), min(len(a), i+w+1)
        out.append(sum(a[lo:hi])/(hi-lo))
    return out
arc = smooth(arc, 2)
arc = [0.45 + 0.55*x for x in arc]   # 0.45..1.0 multiplier

def beat_pos(t):  return (t % BAR) / PPQ          # 0..4
def clamp(v):     return max(1, min(127, int(round(v))))
def hum(amt=4):   return random.uniform(-amt, amt)

# phrase shaping for melodic tracks: 4-bar phrase, rise to bar 3, settle
def phrase_mult(t):
    ph = (t % (4*BAR)) / (4*BAR)                  # 0..1 in phrase
    return 0.85 + 0.25*math.sin(ph*math.pi)      # 0.85 -> 1.10 -> 0.85

def shape(role, notes):
    # detect fast runs (rolls) for crescendo treatment
    for i, x in enumerate(notes):
        a = arc[min(x.t // BAR, n_bars-1)]
        bp = beat_pos(x.t)
        on_beat = abs(bp - round(bp)) < 0.05
        base = 100
        if role == 'kick':
            v = 118 + (6 if bp < 0.05 else 0) + hum(2)
        elif role == 'snare':
            v = 105 + (8 if abs(bp-1)<0.1 or abs(bp-3)<0.1 else 0) + hum(3)
        elif role == 'hat':
            off = abs(bp - math.floor(bp) - 0.5) < 0.1
            v = 72 + (16 if off else 0) + (8 if on_beat else 0) + hum(6)
        elif role == 'perc':
            v = 92 + (12 if on_beat else 0) + (6 if bp < 0.05 else 0) + hum(5)
        elif role == 'bass':
            v = 104 + (10 if bp < 0.05 else 0) + hum(3)
        elif role == 'pad':
            v = 78 * phrase_mult(x.t) + hum(3)
        elif role == 'pluck':
            v = 88 + (10 if i % 2 == 0 else 0) + hum(5)
        elif role == 'guitar':
            v = 98 + (12 if on_beat else 0) + hum(4)
        elif role == 'fx':
            v = 60  # crescendo applied below
        else:  # lead
            rel_pitch = (x.pitch - min(y.pitch for y in notes)) / max(1, (max(y.pitch for y in notes) - min(y.pitch for y in notes)))
            long_note = x.dur > PPQ
            v = 92 * phrase_mult(x.t) + 14*rel_pitch + (8 if long_note else 0) + hum(4)
        x.vel_new = v * a

    # fast-run crescendos (rolls, fx risers): >=6 notes with gaps < 1/8 note
    i = 0
    while i < len(notes):
        j = i
        while j+1 < len(notes) and notes[j+1].t - notes[j].t <= PPQ//2 and notes[j+1].t > notes[j].t: j += 1
        run = j - i + 1
        if run >= 6 and role in ('snare','perc','hat','fx','pluck'):
            for k in range(i, j+1):
                f = (k - i) / max(1, run-1)
                lo_v = notes[i].vel_new * 0.55
                hi_v = min(127, max(notes[j].vel_new, 112))
                notes[k].vel_new = lo_v + (hi_v - lo_v) * (f*f*0.7 + f*0.3)
        i = j + 1

    for x in notes:
        x.msg.velocity = clamp(x.vel_new)

# ---------------- apply ----------------
report = []
for ti in track_info:
    if not ti['notes']: continue
    role = classify(ti['name'], ti['notes'])
    shape(role, ti['notes'])
    vs = [x.msg.velocity for x in ti['notes']]
    report.append(f"  {ti['name'] or '(unnamed)':<20} role={role:<7} notes={len(vs):<6} vel {min(vs)}-{max(vs)}")

m.save(OUT)
print(f"Source: {SRC}\n{n_bars} bars, PPQ {PPQ}\n\nTracks:")
print("\n".join(report))
print(f"\nWrote {OUT}")
