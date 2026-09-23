"""Explicit arranging preferences inferred from one user-edited MIDI example.

This is a readable evidence profile, not model training or a statistical claim
about all of the user's music. It contains no source MIDI or private file paths.
"""
from copy import deepcopy


STYLE_PROFILE = {
    'name': 'Personal phrase arrangement',
    'method': 'Subtractive arrangement with phrase mutes and complementary handoffs',
    'evidence_scope': 'One original song and one user-edited arrangement; inferred preferences, not model training.',
    'reference': {
        'bars': 96,
        'bpm': 110,
        'structure': 'The edit retained the song length and tempo.',
        'melodic_changes': 'Most cuts remove sustained portions of existing guitar, keys and mallet parts.',
        'approximate_note_count_reduction': {'guitar': 0.31, 'keys': 0.54, 'mallet': 0.47},
        'cut_lengths_bars': [1, 2, 4, 8],
        'rhythm_section': 'Drums remained almost unchanged; bass occasionally rests just before an arrival.',
        'chorus_examples': [
            'A sparse first chorus gives the mallet part the lead while guitar and keys rest.',
            'Later choruses change which melodic voice leads.',
            'The final chorus includes a complementary four-bar keys-to-guitar handoff.',
        ],
        'other_phrase_examples': [
            'The intro delays mallet by two bars while guitar opens.',
            'Verse 1 gives keys one-bar rests at the end of each four-bar phrase.',
            'Verse 2 introduces keys in the second half.',
            'Pre-chorus 2 trades guitar for keys halfway through while mallet rests.',
            'Bridge phrases use occasional two-bar gaps in mallet or strings.',
        ],
    },
    'application': [
        'Treat these observations as preferences, not mandatory reduction percentages.',
        'Choose lanes from actual roles, names, pitch ranges and note spans; never reuse example track indices.',
        'Preserve the source notes outside deliberately silent phrases.',
        'Keep a clear melodic foreground and let other parts answer or rest.',
    ],
}


def evidence_profile():
    """Return a fresh profile so callers cannot mutate the shared evidence."""
    return deepcopy(STYLE_PROFILE)


STYLE_INSTRUCTIONS = """
PERSONAL PHRASE ARRANGEMENT (enabled):
The personal_style_profile is evidence from ONE user edit, not model training.
Apply its arranging tendencies to the CURRENT sources, not as a copied song or
fixed mute percentages. Identify musical functions from the actual source names,
roles, ranges and note spans; never hardcode the example's raw track indices or
assume General MIDI programs identify the instruments.

Use subtractive arrangement: phrase-length instrumental dropouts, staggered
entrances, complementary call-and-response and a changing melodic foreground.
Favor intentional 1-, 2-, 4- and 8-bar rests aligned with musical phrases when
the source supports them. One melodic lead at a time is a useful default, while
the chord and rhythm foundation can continue underneath. Avoid simultaneous
clutter from several hook instruments. Build returning-chorus contrast by giving
a different voice the lead or exchanging two voices after four bars. A sparse
early chorus and a later keys-to-guitar-style handoff are examples, not a fixed
template or a demand that those instrument names exist. Explain the decisions.

Each version 3 section requires instrument_cuts, an array of at most 128 objects
{track_index, start_beat, end_beat}. These are SECTION-relative quarter-note
beats, not bars: each repetition starts at source bars * 4 beats. Endpoints must
use the quarter-beat grid, start>=0, end<=section bars*4, duration>=0.25 beat.
Target ONLY active tracks with pitched_note_count>0 and protected=false. NEVER
target Drums, Keep, channel 10 or mixed protected tracks. Use active_tracks for
whole-section rests; use instrument_cuts for phrase rests inside active parts.
Include at least one instrument cut that intersects an existing note span when
any active eligible pitched lane exists. Use pitched_note_spans and activity
metadata to place real cuts, considering all repetitions; do not select silence.

Instrument cuts remove the half-open time interval [start_beat, end_beat) from
the original note spans. Notes crossing the start are shortened. A surviving
tail after the end is retriggered at the cut end with its original pitch and
the same velocity as that original note after permitted dynamics. The gate runs
after dynamics and never applies a new velocity change to a resumed tail. Several
cuts may gate one note into several surviving pieces. No new melody is composed.
Outside these explicit gates and the supplied chord variation, retain source
pitch and timing. Drum cuts retain their separate onset-only semantics.

Keep the drum groove largely intact, matching the example's nearly unchanged
drums. When drum_trimming is on and editable drums exist, use only a restrained
meaningful cut, such as a brief coordinated pre-arrival rest; do not infer broad
intro/verse/bridge drum stripping from the generic full-song guidance. When
drum_trimming is off, drum_cuts must stay empty. Bass usually anchors the groove;
an occasional brief rest before an arrival can support the melodic handoff.
Most subtraction should come from melodic phrase mutes, not repeated drum gaps.
"""
