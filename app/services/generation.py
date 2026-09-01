from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

from app.ml.dataset import build_instruction
from app.ml.model import model_service
from app.ml.parsing import parse_structured_output
from app.schemas.studio import (
    ChordAlignment, GenerateRequest, GenerateResponse, LyricSection, RepairRecord,
    SectionGeneration,
)
from app.services.lyrics import analyze_lyrics, estimate_syllables
from app.theory.engine import SCALE_INTERVALS, theory


CADENCE_BY_SECTION = {"Intro": "weak", "Verse": "half", "Pre-Chorus": "half", "Chorus": "strong",
                      "Hook": "strong", "Bridge": "deceptive", "Breakdown": "plagal", "Outro": "plagal"}
BASE_PROGRESSIONS = {
    "strong": [
        ["I", "V", "vi", "IV", "V", "I"], ["I", "IV", "ii", "V", "I"],
        ["vi", "IV", "I", "iii", "V", "I"], ["I", "iii", "vi", "ii", "V", "I"],
        ["I", "vi", "IV", "ii", "V", "I"], ["I", "IV", "vi", "iii", "V", "I"],
    ],
    "half": [
        ["I", "vi", "IV", "ii", "V"], ["vi", "IV", "I", "ii", "V"],
        ["I", "iii", "IV", "vi", "V"], ["iii", "vi", "IV", "ii", "V"],
        ["I", "IV", "I", "ii", "V"], ["vi", "iii", "IV", "ii", "V"],
    ],
    "deceptive": [
        ["I", "IV", "ii", "V", "vi"], ["vi", "ii", "IV", "V", "vi"],
        ["I", "iii", "IV", "V", "vi"], ["I", "vi", "ii", "V", "vi"],
        ["IV", "I", "ii", "V", "vi"], ["I", "IV", "iii", "V", "vi"],
    ],
    "plagal": [
        ["I", "vi", "IV", "I"], ["vi", "ii", "IV", "I"],
        ["I", "iii", "IV", "I"], ["vi", "I", "IV", "I"],
        ["I", "ii", "IV", "I"], ["iii", "vi", "IV", "I"],
    ],
    "weak": [
        ["I", "iii", "vi", "IV"], ["vi", "IV", "I", "V"],
        ["I", "ii", "vi", "IV"], ["iii", "IV", "I", "ii"],
        ["vi", "iii", "IV", "I"], ["I", "IV", "iii", "vi"],
    ],
}

MOOD_TOKENS = {
    "happy": "IV", "hopeful": "vi", "sad": "vi", "dark": "iv",
    "calm": "iii", "energetic": "V", "romantic": "ii", "reflective": "vi",
}
GENRE_TOKENS = {
    "pop": "vi", "rock": "IV", "acoustic": "ii", "ballad": "iii",
    "folk": "IV", "blues": "IV7", "r&b": "ii7", "cinematic": "bVI",
}
DECODING_OFFSETS = {"greedy": 0, "beam": 1, "temperature": 2, "top_k": 3}


def _fallback(section: LyricSection, request: GenerateRequest, section_index: int) -> SectionGeneration:
    controls = request.controls
    cadence = CADENCE_BY_SECTION.get(section.name, "weak")
    candidates = BASE_PROGRESSIONS[cadence]
    signature = json.dumps({
        "section": section.name, "section_index": section_index, "lyrics": section.lines,
        "genre": controls.genre.lower(), "mood": controls.mood.lower(), "tempo_bucket": controls.tempo // 15,
        "time_signature": controls.time_signature, "difficulty": controls.difficulty,
        "density": controls.chord_density, "variation_bucket": round(controls.variation * 10),
        "decoding": controls.decoding_method, "previous": request.previous_progression,
    }, sort_keys=True, ensure_ascii=False)
    stable = int(hashlib.sha256(signature.encode()).hexdigest()[:12], 16)
    candidate_index = (stable + controls.seed + DECODING_OFFSETS[controls.decoding_method]) % len(candidates)
    progression = list(candidates[candidate_index])
    # Mood and genre alter a middle harmonic color while preserving the section cadence.
    color_index = 1 + (stable % max(1, len(progression) - 3))
    if controls.variation >= 0.25:
        mood_token = MOOD_TOKENS.get(controls.mood.lower(), progression[color_index])
        progression[color_index] = "IV" if controls.difficulty == "beginner" and mood_token == "iv" else mood_token
    if controls.variation >= 0.55:
        genre_index = 1 + ((stable // 17) % max(1, len(progression) - 3))
        genre_token = GENRE_TOKENS.get(controls.genre.lower(), progression[genre_index])
        if controls.difficulty == "beginner":
            genre_token = "vi" if genre_token.startswith(("b", "#")) else re.sub(r"7$", "", genre_token)
        progression[genre_index] = genre_token
    if controls.tempo >= 130 and len(progression) > 4:
        progression.insert(-2, "V")
    elif controls.tempo <= 70 and len(progression) > 4:
        progression.pop(1 + stable % (len(progression) - 3))

    if controls.difficulty == "intermediate":
        index = 1 + (stable % max(1, len(progression) - 3))
        progression[index] = re.sub(r"(7|sus4|6)$", "", progression[index]) + ["7", "sus4", "6"][stable % 3]
    elif controls.difficulty == "advanced":
        index = 1 + (stable % max(1, len(progression) - 3))
        progression[index] = ["V7/V", "iv", "bVII", "#iv°7", "ii65", "V9"][stable % 6]
    if controls.variation > 0.8 and len(progression) > 4:
        first, second = 1, 2 + (stable % max(1, len(progression) - 3))
        progression[first], progression[second] = progression[second], progression[first]
    if progression == request.previous_progression and len(candidates) > 1:
        progression = list(candidates[(candidate_index + 1) % len(candidates)])
    progression, _ = theory.constrain_progression(progression, controls.difficulty, cadence)
    numerator = int(controls.time_signature.split("/")[0])
    beats = [float(max(1, numerator / controls.chord_density))] * len(progression)
    return SectionGeneration(
        section_name=section.name, roman_numerals=progression, beats_per_chord=beats,
        cadence_type=cadence, energy_level="high" if controls.tempo > 120 else "low" if controls.tempo < 75 else "medium",
        confidence_notes="Deterministic theory fallback; no neural-model claim is made.",
    )


def _instruction(section: LyricSection, request: GenerateRequest, previous: list[str]) -> str:
    features = {
        "lines": [feature.model_dump() for feature in section.features],
        "max_suggested_changes": max((feature.suggested_chord_changes for feature in section.features), default=1),
    }
    controls = request.controls
    return build_instruction({
        "lyrics": "\n".join(section.lines), "section": section.name, "key": controls.key, "scale": controls.scale,
        "genre": controls.genre, "mood": controls.mood, "tempo": controls.tempo,
        "time_signature": controls.time_signature, "difficulty": controls.difficulty,
        "chord_density": controls.chord_density, "variation": controls.variation,
        "previous_progression": previous, "lyric_features": features,
    })


def _align(section: LyricSection, generated: SectionGeneration, key: str, scale: str, chord_density: int) -> list[ChordAlignment]:
    alignments: list[ChordAlignment] = []
    chord_index = 0
    for line_index, line in enumerate(section.lines):
        words = line.split()
        feature_words = section.features[line_index].word_count if line_index < len(section.features) else len(words)
        desired = round(chord_density * min(1.0, max(1, feature_words) / 8))
        desired = max(1, min(desired, len(words) or 1, len(generated.roman_numerals)))
        positions = sorted({min(max(0, len(words) - 1), round(i * max(1, len(words) - 1) / desired)) for i in range(desired)})
        for word_index in positions:
            roman = generated.roman_numerals[chord_index % len(generated.roman_numerals)]
            rendered = theory.render_roman(roman, key, scale)
            beats = generated.beats_per_chord[chord_index % len(generated.beats_per_chord)]
            alignments.append(ChordAlignment(
                section_name=section.name, line_index=line_index, word_index=word_index,
                roman_numeral=roman, chord=rendered.symbol, beats=beats,
            ))
            chord_index += 1
    return alignments


def generate_song(request: GenerateRequest, only_section: str | None = None) -> GenerateResponse:
    started = time.perf_counter()
    analysis = analyze_lyrics(request.lyrics) if request.sections is None else analyze_lyrics(request.lyrics)
    if request.sections is not None:
        analysis.sections = request.sections
        analysis.total_lines = sum(len(section.lines) for section in request.sections)
    generated_sections: list[SectionGeneration] = []
    raw_outputs: dict[str, str] = {}
    repairs: list[RepairRecord] = []
    alignments: list[ChordAlignment] = []
    sources: list[str] = []
    previous = request.previous_progression
    model_latency = 0.0
    for index, section in enumerate(analysis.sections):
        if only_section and section.name.lower() != only_section.lower():
            continue
        instruction = _instruction(section, request, previous)
        raw, latency = model_service.generate(instruction, request.controls)
        model_latency += latency
        parsed = None
        parse_repaired = False
        parse_error = None
        if raw is not None:
            parsed, parse_repaired, parse_error = parse_structured_output(raw)
            raw_outputs[section.name] = raw
        if parsed is None:
            parsed = _fallback(section, request, index)
            source = "algorithmic_fallback_output"
            reasons = ["Neural output unavailable or failed structured validation"]
            if parse_error: reasons.append(parse_error)
        else:
            source = "lora_model_output" if model_service.adapter_loaded else "base_model_output"
            reasons = ["Deterministic JSON parsing repair applied"] if parse_repaired else []
            if parsed.section_name != section.name:
                reasons.append(f"Corrected section name from {parsed.section_name} to {section.name}")
                parsed.section_name = section.name
        original = list(parsed.roman_numerals)
        desired_cadence = parsed.cadence_type if parsed.cadence_type in {"strong", "weak", "deceptive", "half", "plagal"} else CADENCE_BY_SECTION.get(section.name, "weak")
        constrained, constraint_reasons = theory.constrain_progression(original, request.controls.difficulty, desired_cadence)
        reasons.extend(constraint_reasons)
        parsed.roman_numerals = constrained
        if len(parsed.beats_per_chord) != len(constrained):
            beat = parsed.beats_per_chord[0] if parsed.beats_per_chord else float(int(request.controls.time_signature.split("/")[0]))
            parsed.beats_per_chord = [beat] * len(constrained)
            reasons.append("Normalized beat durations to match the chord count")
        if reasons and source == "lora_model_output": source = "lora_output_repaired_by_constraints"
        repairs.append(RepairRecord(section_name=section.name, reasons=list(dict.fromkeys(reasons)), original=original, repaired=constrained))
        generated_sections.append(parsed)
        alignments.extend(_align(section, parsed, request.controls.key, request.controls.scale, request.controls.chord_density))
        previous = constrained
        sources.append(source)
    overall = sources[0] if sources and len(set(sources)) == 1 else "+".join(sorted(set(sources)))
    total_latency = model_latency or (time.perf_counter() - started) * 1000
    return GenerateResponse(
        project_name=request.project_name, analysis=analysis, sections=generated_sections, alignments=alignments,
        raw_model_outputs=raw_outputs, repairs=repairs, generation_source=overall,
        inference_latency_ms=round(total_latency, 3), controls=request.controls,
    )


def transpose_result(result: GenerateResponse, target_key: str) -> GenerateResponse:
    source_key = result.controls.key
    semitones = theory.key_distance(source_key, target_key)
    prefer_flats = "b" in target_key
    for alignment in result.alignments:
        alignment.chord = theory.transpose_symbol(alignment.chord, semitones, prefer_flats)
    result.controls.key = theory.normalize_key(target_key)
    return result


def _melody_for_alignment(
    result: GenerateResponse,
    alignment: ChordAlignment,
    next_alignment: ChordAlignment | None,
) -> tuple[list[int], float, str]:
    section = next((item for item in result.analysis.sections if item.name == alignment.section_name), None)
    if section is None or alignment.line_index >= len(section.lines):
        return [], alignment.beats, ""
    line = section.lines[alignment.line_index]
    words = line.split()
    start = min(alignment.word_index, max(0, len(words) - 1))
    end = len(words)
    if (
        next_alignment is not None
        and next_alignment.section_name == alignment.section_name
        and next_alignment.line_index == alignment.line_index
        and next_alignment.word_index > start
    ):
        end = next_alignment.word_index
    fragment_words = words[start:end] or words[start:start + 1]
    fragment = " ".join(fragment_words)
    syllables = sum(estimate_syllables(word) for word in fragment_words)
    note_count = max(1, min(8, syllables))

    all_lyrics = "\n".join(
        line_text
        for analysis_section in result.analysis.sections
        for line_text in analysis_section.lines
    )
    controls = result.controls
    signature = "|".join([
        all_lyrics, alignment.section_name, str(alignment.line_index), str(alignment.word_index),
        fragment, alignment.chord, controls.key, controls.scale, controls.genre, controls.mood,
        str(controls.tempo), str(controls.variation), str(controls.seed),
    ])
    digest = hashlib.sha256(signature.encode("utf-8")).digest()
    tonic_pc = theory.pitch_class(controls.key)
    scale_intervals = SCALE_INTERVALS[controls.scale]
    scale_pcs = [(tonic_pc + interval) % 12 for interval in scale_intervals]
    chord_pcs = {note % 12 for note in theory.chord_to_midi(alignment.chord)}
    chord_degrees = [index for index, pc in enumerate(scale_pcs) if pc in chord_pcs]
    degree = chord_degrees[digest[0] % len(chord_degrees)] if chord_degrees else digest[0] % len(scale_intervals)
    feature = section.features[alignment.line_index] if alignment.line_index < len(section.features) else None
    emotion = feature.emotional_word_score if feature else 0.0
    mood_direction = 1 if controls.mood.lower() in {"happy", "hopeful", "energetic", "romantic"} else -1 if controls.mood.lower() in {"sad", "dark", "reflective"} else 0
    contour_bias = mood_direction + (1 if emotion > 0.05 else -1 if emotion < -0.05 else 0)
    step_choices = (-2, -1, 0, 1, 2)
    melody: list[int] = []
    for index in range(note_count):
        if index:
            raw_step = step_choices[digest[(index + 1) % len(digest)] % len(step_choices)]
            if controls.variation < 0.3:
                raw_step = max(-1, min(1, raw_step))
            degree = max(0, min(len(scale_intervals) - 1, degree + raw_step + (contour_bias if index == 1 else 0)))
        note = 72 + tonic_pc + scale_intervals[degree]
        while note > 88:
            note -= 12
        while note < 64:
            note += 12
        melody.append(note)
    if fragment.rstrip().endswith((".", "!", "?")) and melody:
        chord_targets = [note for note in theory.chord_to_midi(alignment.chord, octave=5) if 64 <= note <= 88]
        if chord_targets:
            melody[-1] = min(chord_targets, key=lambda note: abs(note - melody[-1]))
    return melody, round(alignment.beats / note_count, 4), fragment[:240]


def playback_timeline(result: GenerateResponse) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    beat = 0.0
    for index, alignment in enumerate(result.alignments):
        notes = theory.chord_to_midi(alignment.chord)
        next_alignment = result.alignments[index + 1] if index + 1 < len(result.alignments) else None
        melody, melody_note_beats, fragment = _melody_for_alignment(result, alignment, next_alignment)
        events.append({
            "section_name": alignment.section_name, "line_index": alignment.line_index,
            "word_index": alignment.word_index, "chord": alignment.chord, "midi_notes": notes,
            "melody_midi_notes": melody, "melody_note_beats": melody_note_beats,
            "lyric_fragment": fragment, "start_beat": beat,
            "duration_beats": alignment.beats,
        })
        beat += alignment.beats
    return events
