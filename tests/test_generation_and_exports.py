from __future__ import annotations

import io
import json
from copy import deepcopy

from mido import MidiFile

from app.schemas.studio import GenerateRequest
from app.services.exports import chord_sheet_text, json_export, midi_export, pdf_export
from app.services.generation import generate_song, playback_timeline, transpose_result


def music_fingerprint(result):
    return json.dumps({
        "sections": [{"roman": section.roman_numerals, "beats": section.beats_per_chord} for section in result.sections],
        "alignment": [(item.section_name, item.line_index, item.word_index, item.chord, item.beats) for item in result.alignments],
    }, sort_keys=True)


def test_deterministic_fallback_and_seed_variation(monkeypatch, song_payload):
    from app.ml.model import model_service
    monkeypatch.setattr(model_service, "generate", lambda instruction, controls: (None, 0.0))
    first = generate_song(GenerateRequest.model_validate(song_payload)); second = generate_song(GenerateRequest.model_validate(song_payload))
    assert music_fingerprint(first) == music_fingerprint(second)
    changed = deepcopy(song_payload); changed["controls"]["seed"] = 43
    variation = generate_song(GenerateRequest.model_validate(changed))
    assert first.generation_source == "algorithmic_fallback_output"
    assert music_fingerprint(first) != music_fingerprint(variation)


def test_fallback_responds_to_musical_controls_and_lyrics(monkeypatch, song_payload):
    from app.ml.model import model_service
    monkeypatch.setattr(model_service, "generate", lambda instruction, controls: (None, 0.0))
    base = generate_song(GenerateRequest.model_validate(song_payload))
    base_fingerprint = music_fingerprint(base)
    variants = []
    for field, value in [
        ("mood", "dark"), ("genre", "cinematic"), ("tempo", 150),
        ("chord_density", 4), ("variation", .9), ("decoding_method", "beam"),
    ]:
        changed = deepcopy(song_payload); changed["controls"][field] = value
        variants.append(generate_song(GenerateRequest.model_validate(changed)))
    changed_lyrics = deepcopy(song_payload)
    changed_lyrics["lyrics"] = "[Verse]\nNight gathers slowly over the quiet road\n[Chorus]\nA distant answer rises with the tide"
    variants.append(generate_song(GenerateRequest.model_validate(changed_lyrics)))
    assert all(music_fingerprint(result) != base_fingerprint for result in variants)
    dense = variants[3]
    assert len(dense.alignments) > len(base.alignments)


def test_transposition_timeline_and_exports(monkeypatch, song_payload):
    from app.ml.model import model_service
    monkeypatch.setattr(model_service, "generate", lambda instruction, controls: (None, 0.0))
    result = generate_song(GenerateRequest.model_validate(song_payload))
    original = result.alignments[0].chord
    transposed = transpose_result(result.model_copy(deep=True), "D")
    assert transposed.controls.key == "D" and transposed.alignments[0].chord != original
    timeline = playback_timeline(transposed)
    assert timeline and timeline[0]["midi_notes"] and timeline[0]["melody_midi_notes"]
    assert timeline[0]["lyric_fragment"] and timeline[0]["melody_note_beats"] > 0
    assert timeline[-1]["start_beat"] >= timeline[0]["start_beat"]
    assert "Test Song" in chord_sheet_text(transposed)
    assert json.loads(json_export(transposed))["controls"]["key"] == "D"
    assert pdf_export(transposed).startswith(b"%PDF")
    midi_bytes = midi_export(transposed)
    assert midi_bytes.startswith(b"MThd")
    assert len(MidiFile(file=io.BytesIO(midi_bytes)).tracks) == 3


def test_lyric_changes_create_a_distinct_playback_melody(monkeypatch, song_payload):
    from app.ml.model import model_service
    monkeypatch.setattr(model_service, "generate", lambda instruction, controls: (None, 0.0))
    first = generate_song(GenerateRequest.model_validate(song_payload))
    changed = deepcopy(song_payload)
    changed["lyrics"] = "[Verse]\nEvery night the ocean carries a distant dream\n[Chorus]\nOur promise sails beyond the fading stars"
    second = generate_song(GenerateRequest.model_validate(changed))
    first_melody = [note for event in playback_timeline(first) for note in event["melody_midi_notes"]]
    second_melody = [note for event in playback_timeline(second) for note in event["melody_midi_notes"]]
    assert first_melody and second_melody and first_melody != second_melody
    assert all(64 <= note <= 88 for note in first_melody + second_melody)
