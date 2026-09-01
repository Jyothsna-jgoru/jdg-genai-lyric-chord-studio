import pytest

from app.theory.engine import theory


def test_scale_and_enharmonic_spelling():
    assert theory.spell_scale("F#", "major") == ["F#", "G#", "A#", "B", "C#", "D#", "E#"]
    assert theory.spell_scale("Bb", "major") == ["Bb", "C", "D", "Eb", "F", "G", "A"]


def test_roman_rendering_and_borrowed_chord():
    assert theory.render_roman("V7", "C").symbol == "G7"
    assert theory.render_roman("ii6", "C").symbol == "Dm/F"
    assert theory.render_roman("bVII", "C").symbol == "Bb"
    assert theory.render_roman("V7/V", "C").symbol == "D7"


def test_cadences_and_constraints():
    assert theory.cadence_type(["ii", "V", "I"]) == "strong"
    repaired, reasons = theory.constrain_progression(["Imaj7", "bad", "V9", "I"], "beginner", "strong")
    assert all(theory.is_valid_roman(token) for token in repaired)
    assert theory.cadence_type(repaired) == "strong"
    assert reasons


def test_chords_midi_voice_leading_and_validation():
    assert theory.chord_to_midi("C") == [60, 64, 67]
    assert theory.chord_to_midi("Ebm7") == [63, 66, 70, 73]
    assert len(theory.chord_to_midi("G7/B")) == 5
    valid, _ = theory.validate_chord("Ebm7", "intermediate"); assert valid
    valid, _ = theory.validate_chord("C", "beginner"); assert valid
    valid, reason = theory.validate_chord("Cmaj7", "beginner"); assert not valid and "beginner" in reason
    chords = [theory.render_roman(token, "C") for token in ["I", "IV", "V", "I"]]
    assert 0 < theory.voice_leading_score(chords) <= 1


def test_transposition():
    assert theory.key_distance("C", "D") == 2
    assert theory.transpose_symbol("Cmaj7/E", 2) == "Dmaj7/F#"
    with pytest.raises(ValueError): theory.transpose_symbol("H7", 2)
