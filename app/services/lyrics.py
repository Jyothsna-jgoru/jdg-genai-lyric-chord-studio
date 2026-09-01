from __future__ import annotations

import re
from collections import Counter

from app.schemas.studio import LineFeature, LyricAnalysisResponse, LyricSection


SECTION_NAMES = {
    "intro": "Intro",
    "verse": "Verse",
    "pre-chorus": "Pre-Chorus",
    "pre chorus": "Pre-Chorus",
    "chorus": "Chorus",
    "hook": "Hook",
    "bridge": "Bridge",
    "breakdown": "Breakdown",
    "outro": "Outro",
}
EMOTION_LEXICON = {
    "love": 1.0, "hope": 0.9, "bright": 0.8, "joy": 1.0, "smile": 0.8,
    "free": 0.7, "dream": 0.6, "home": 0.5, "warm": 0.6, "rise": 0.6,
    "sad": -0.9, "cry": -0.9, "lost": -0.8, "dark": -0.7, "alone": -0.8,
    "fear": -0.8, "cold": -0.6, "fall": -0.5, "broken": -1.0, "goodbye": -0.7,
}
VOWEL_GROUP = re.compile(r"[aeiouy]+", re.I)
WORD = re.compile(r"[A-Za-z']+")
HEADING = re.compile(
    r"^\s*(?:\[|\(|\{)?\s*(intro|verse(?:\s+\d+)?|pre[- ]chorus|chorus(?:\s+\d+)?|hook|bridge|breakdown|outro)\s*(?:\]|\)|\}|:)?\s*$",
    re.I,
)


def estimate_syllables(word: str) -> int:
    clean = re.sub(r"[^a-z]", "", word.lower())
    if not clean:
        return 0
    groups = len(VOWEL_GROUP.findall(clean))
    if clean.endswith("e") and not clean.endswith(("le", "ye")) and groups > 1:
        groups -= 1
    if clean.endswith("ed") and len(clean) > 3 and not clean.endswith(("ted", "ded")) and groups > 1:
        groups -= 1
    return max(1, groups)


def _heading_name(raw: str) -> str | None:
    match = HEADING.match(raw)
    if not match:
        return None
    token = re.sub(r"\s+\d+$", "", match.group(1).lower())
    return SECTION_NAMES.get(token.replace("-", " "), SECTION_NAMES.get(token, token.title()))


def analyze_lyrics(lyrics: str) -> LyricAnalysisResponse:
    raw_lines = lyrics.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    sections: list[tuple[str, list[str]]] = []
    current_name = "Verse"
    current_lines: list[str] = []
    heading_count = 0
    for raw in raw_lines:
        heading = _heading_name(raw)
        if heading:
            if current_lines:
                sections.append((current_name, current_lines))
            current_name, current_lines = heading, []
            heading_count += 1
        elif raw.strip():
            current_lines.append(raw.rstrip())
    if current_lines:
        sections.append((current_name, current_lines))
    if not sections:
        sections = [("Verse", [lyrics.strip()])]

    normalized = [re.sub(r"\s+", " ", line.strip().lower()) for _, lines in sections for line in lines]
    counts = Counter(normalized)
    total_sections = len(sections)
    result: list[LyricSection] = []
    for section_index, (name, lines) in enumerate(sections):
        features: list[LineFeature] = []
        for line_index, line in enumerate(lines):
            words = WORD.findall(line)
            syllables = sum(estimate_syllables(word) for word in words)
            score = sum(EMOTION_LEXICON.get(word.lower(), 0.0) for word in words)
            score = round(score / max(1, len(words)), 3)
            changes = max(1, min(4, round(max(1, len(words)) / 4)))
            punctuation = "".join(character for character in line if character in ".,!?;:")
            density = "low" if syllables <= 6 else "medium" if syllables <= 12 else "high"
            features.append(LineFeature(
                text=line,
                word_count=len(words),
                syllable_count=syllables,
                repeated=counts[re.sub(r"\s+", " ", line.strip().lower())] > 1,
                punctuation=punctuation,
                section_position=round(section_index / max(1, total_sections - 1), 3),
                line_position=round(line_index / max(1, len(lines) - 1), 3),
                emotional_word_score=score,
                suggested_chord_changes=changes,
                suggested_rhythmic_density=density,
            ))
        result.append(LyricSection(name=name, lines=lines, features=features))
    return LyricAnalysisResponse(sections=result, heading_count=heading_count, total_lines=sum(len(s.lines) for s in result))

