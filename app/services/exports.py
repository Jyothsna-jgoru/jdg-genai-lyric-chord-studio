from __future__ import annotations

import io
import json
import re
from collections import defaultdict

import mido
from mido import Message, MetaMessage, MidiFile, MidiTrack
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from app.schemas.studio import GenerateResponse
from app.services.generation import playback_timeline
from app.theory.engine import theory


def safe_filename(name: str, extension: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-.")[:80] or "song"
    return f"{clean}.{extension}"


def chord_sheet_text(result: GenerateResponse) -> str:
    lines = [result.project_name, f"Key: {result.controls.key} | Scale: {result.controls.scale} | Tempo: {result.controls.tempo} BPM", ""]
    grouped = defaultdict(list)
    for alignment in result.alignments: grouped[(alignment.section_name, alignment.line_index)].append(alignment)
    for section in result.analysis.sections:
        lines.append(f"[{section.name}]")
        for index, lyric in enumerate(section.lines):
            words = lyric.split()
            chord_line = [" " * len(word) for word in words]
            for item in grouped.get((section.name, index), []):
                if words:
                    chord_line[min(item.word_index, len(words) - 1)] = item.chord
            lines.append(" ".join(chord_line).rstrip())
            lines.append(lyric)
        lines.append("")
    return "\n".join(lines)


def json_export(result: GenerateResponse) -> bytes:
    return json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False).encode("utf-8")


def pdf_export(result: GenerateResponse) -> bytes:
    buffer = io.BytesIO(); styles = getSampleStyleSheet()
    document = SimpleDocTemplate(buffer, pagesize=letter, title=result.project_name, author="Jyothsna Devi Goru")
    story = [Paragraph(result.project_name, styles["Title"]),
             Paragraph(f"Key: {result.controls.key} &nbsp; Tempo: {result.controls.tempo} BPM &nbsp; Time: {result.controls.time_signature}", styles["Normal"]), Spacer(1, 12)]
    for block in chord_sheet_text(result).split("\n\n")[1:]:
        escaped = block.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
        story.extend([Paragraph(escaped, styles["Code"]), Spacer(1, 10)])
    document.build(story)
    return buffer.getvalue()


def midi_export(result: GenerateResponse) -> bytes:
    midi = MidiFile(type=1, ticks_per_beat=480)
    meta = MidiTrack(); midi.tracks.append(meta)
    meta.append(MetaMessage("track_name", name=result.project_name, time=0))
    meta.append(MetaMessage("set_tempo", tempo=mido.bpm2tempo(result.controls.tempo), time=0))
    numerator, denominator = map(int, result.controls.time_signature.split("/"))
    meta.append(MetaMessage("time_signature", numerator=numerator, denominator=denominator, time=0))
    track = MidiTrack(); midi.tracks.append(track)
    current_section = None
    for item in result.alignments:
        if item.section_name != current_section:
            current_section = item.section_name
            track.append(MetaMessage("marker", text=current_section, time=0))
        notes = theory.chord_to_midi(item.chord)
        for note in notes: track.append(Message("note_on", note=note, velocity=64, time=0))
        duration = max(1, round(item.beats * midi.ticks_per_beat))
        for note_index, note in enumerate(notes):
            track.append(Message("note_off", note=note, velocity=0, time=duration if note_index == 0 else 0))
    melody_track = MidiTrack(); midi.tracks.append(melody_track)
    melody_track.append(MetaMessage("track_name", name="Lyric-shaped melody", time=0))
    current_section = None
    for event in playback_timeline(result):
        if event["section_name"] != current_section:
            current_section = event["section_name"]
            melody_track.append(MetaMessage("marker", text=current_section, time=0))
        duration = max(1, round(event["melody_note_beats"] * midi.ticks_per_beat))
        for note in event["melody_midi_notes"]:
            melody_track.append(Message("note_on", note=note, velocity=82, time=0))
            melody_track.append(Message("note_off", note=note, velocity=0, time=duration))
    buffer = io.BytesIO(); midi.save(file=buffer); return buffer.getvalue()
