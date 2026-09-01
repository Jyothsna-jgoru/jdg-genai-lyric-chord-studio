from app.services.lyrics import analyze_lyrics, estimate_syllables


def test_section_headings_and_features():
    result = analyze_lyrics("(Intro)\nSoft rain\nVerse 1:\nLove is here\n[Pre-Chorus]\nWe rise\nChorus\nLove is here\nOutro:\nHome")
    assert [section.name for section in result.sections] == ["Intro", "Verse", "Pre-Chorus", "Chorus", "Outro"]
    assert result.heading_count == 5
    assert result.sections[1].features[0].emotional_word_score > 0
    assert result.sections[1].features[0].repeated is True


def test_missing_heading_becomes_verse():
    result = analyze_lyrics("A single original line")
    assert result.sections[0].name == "Verse"


def test_syllable_estimation():
    assert estimate_syllables("morning") == 2
    assert estimate_syllables("time") == 1
    assert estimate_syllables("rhythm") >= 1

