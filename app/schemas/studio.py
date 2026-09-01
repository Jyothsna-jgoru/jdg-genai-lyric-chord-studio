from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


ScaleName = Literal["major", "natural_minor", "harmonic_minor", "melodic_minor"]
Difficulty = Literal["beginner", "intermediate", "advanced"]
DecodingMethod = Literal["greedy", "beam", "temperature", "top_k"]


class LineFeature(BaseModel):
    text: str
    word_count: int
    syllable_count: int
    repeated: bool
    punctuation: str
    section_position: float
    line_position: float
    emotional_word_score: float
    suggested_chord_changes: int
    suggested_rhythmic_density: str


class LyricSection(BaseModel):
    name: str
    lines: list[str]
    features: list[LineFeature] = Field(default_factory=list)


class LyricAnalysisRequest(BaseModel):
    lyrics: str = Field(min_length=1, max_length=20000)

    @field_validator("lyrics")
    @classmethod
    def non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Lyrics cannot be empty")
        return value


class LyricAnalysisResponse(BaseModel):
    sections: list[LyricSection]
    heading_count: int
    total_lines: int


class MusicalControls(BaseModel):
    key: str = "C"
    scale: ScaleName = "major"
    genre: str = "pop"
    mood: str = "hopeful"
    tempo: int = Field(default=96, ge=30, le=240)
    time_signature: str = Field(default="4/4", pattern=r"^(2|3|4|5|6|7|9|12)/(2|4|8)$")
    difficulty: Difficulty = "beginner"
    chord_density: int = Field(default=2, ge=1, le=4)
    variation: float = Field(default=0.35, ge=0, le=1)
    seed: int = Field(default=42, ge=0, le=2147483647)
    decoding_method: DecodingMethod = "greedy"
    num_beams: int = Field(default=4, ge=1, le=8)
    temperature: float = Field(default=0.8, ge=0.1, le=2)
    top_k: int = Field(default=40, ge=1, le=100)


class SectionGeneration(BaseModel):
    section_name: str
    roman_numerals: list[str] = Field(min_length=1, max_length=32)
    beats_per_chord: list[float] = Field(min_length=1, max_length=32)
    cadence_type: str
    energy_level: str
    confidence_notes: str

    @field_validator("beats_per_chord")
    @classmethod
    def positive_beats(cls, values: list[float]) -> list[float]:
        if any(value <= 0 or value > 32 for value in values):
            raise ValueError("beats_per_chord entries must be between 0 and 32")
        return values


class ChordAlignment(BaseModel):
    section_name: str
    line_index: int
    word_index: int
    roman_numeral: str
    chord: str
    beats: float


class RepairRecord(BaseModel):
    section_name: str
    reasons: list[str] = Field(default_factory=list)
    original: list[str] = Field(default_factory=list)
    repaired: list[str] = Field(default_factory=list)


class GenerateRequest(BaseModel):
    project_name: str = Field(default="Untitled Song", min_length=1, max_length=120)
    lyrics: str = Field(min_length=1, max_length=20000)
    sections: list[LyricSection] | None = None
    controls: MusicalControls = Field(default_factory=MusicalControls)
    previous_progression: list[str] = Field(default_factory=list)

    @field_validator("project_name")
    @classmethod
    def safe_name(cls, value: str) -> str:
        cleaned = " ".join(value.replace("<", "").replace(">", "").split())
        if not cleaned:
            raise ValueError("Project name cannot be blank")
        return cleaned


class GenerateResponse(BaseModel):
    project_name: str
    analysis: LyricAnalysisResponse
    sections: list[SectionGeneration]
    alignments: list[ChordAlignment]
    raw_model_outputs: dict[str, str]
    repairs: list[RepairRecord]
    generation_source: str
    inference_latency_ms: float
    controls: MusicalControls


class RegenerateRequest(GenerateRequest):
    section_name: str


class ValidateRequest(BaseModel):
    chord: str
    key: str = "C"
    scale: ScaleName = "major"
    difficulty: Difficulty = "beginner"


class TransposeRequest(BaseModel):
    result: GenerateResponse
    target_key: str


class TimelineEvent(BaseModel):
    section_name: str
    line_index: int
    word_index: int
    chord: str
    midi_notes: list[int]
    melody_midi_notes: list[int] = Field(default_factory=list)
    melody_note_beats: float = Field(default=0.5, gt=0)
    lyric_fragment: str = ""
    start_beat: float
    duration_beats: float


class TimelineRequest(BaseModel):
    result: GenerateResponse


class ProjectPayload(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    lyrics: str = Field(min_length=1, max_length=20000)
    result: GenerateResponse
    manual_chord_edits: list[dict[str, Any]] = Field(default_factory=list)


class ProjectRename(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class ProjectSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    generation_source: str
    created_at: Any
    updated_at: Any


class ProjectDetail(ProjectSummary):
    lyrics: str
    result: dict[str, Any]
    manual_chord_edits: list[dict[str, Any]]
