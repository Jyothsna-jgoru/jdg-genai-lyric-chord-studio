from __future__ import annotations

import math
import re
from dataclasses import dataclass


NOTE_PC = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "FB": 4,
           "E#": 5, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9,
           "A#": 10, "BB": 10, "B": 11, "CB": 11, "B#": 0}
SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
LETTERS = ["C", "D", "E", "F", "G", "A", "B"]
NATURAL_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SCALE_INTERVALS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "natural_minor": [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
}
DEGREES = {"I": 0, "II": 1, "III": 2, "IV": 3, "V": 4, "VI": 5, "VII": 6}
FLAT_KEYS = {"F", "BB", "EB", "AB", "DB", "GB", "CB", "DM", "GM", "CM", "FM", "BBM", "EBM"}


@dataclass(frozen=True)
class ParsedRoman:
    token: str
    degree: int
    accidental: int
    uppercase: bool
    quality: str
    extension: str
    inversion: str
    secondary_degree: int | None = None


@dataclass(frozen=True)
class RenderedChord:
    roman: str
    symbol: str
    midi_notes: list[int]
    pitch_classes: list[int]


class TheoryEngine:
    roman_pattern = re.compile(
        r"^(?P<acc>[b#]*)(?P<roman>VII|III|VI|IV|II|V|I|vii|iii|vi|iv|ii|v|i)"
        r"(?P<quality>min|dim|aug|m|\+|°|ø)?(?P<ext>maj7|add9|sus2|sus4|13|11|9|7)?"
        r"(?P<inv>64|65|43|42|6)?$"
    )

    def normalize_key(self, key: str) -> str:
        key = key.strip().replace("♭", "b").replace("♯", "#")
        if not re.fullmatch(r"[A-Ga-g](?:#|b)?", key):
            raise ValueError(f"Unsupported key: {key}")
        return key[0].upper() + key[1:]

    def pitch_class(self, note: str) -> int:
        normalized = note.strip().replace("♭", "b").replace("♯", "#").upper()
        if normalized not in NOTE_PC:
            raise ValueError(f"Unsupported note: {note}")
        return NOTE_PC[normalized]

    def spell_scale(self, key: str, scale: str = "major") -> list[str]:
        key = self.normalize_key(key)
        if scale not in SCALE_INTERVALS:
            raise ValueError(f"Unsupported scale: {scale}")
        tonic_pc = self.pitch_class(key)
        start = LETTERS.index(key[0])
        notes: list[str] = []
        for index, interval in enumerate(SCALE_INTERVALS[scale]):
            letter = LETTERS[(start + index) % 7]
            desired = (tonic_pc + interval) % 12
            delta = (desired - NATURAL_PC[letter]) % 12
            accidental = "" if delta == 0 else "#" if delta == 1 else "b" if delta == 11 else "##" if delta == 2 else "bb"
            notes.append(letter + accidental)
        return notes

    def parse_roman(self, token: str) -> ParsedRoman:
        clean = token.strip().replace("♭", "b").replace("♯", "#")
        parts = clean.split("/")
        if len(parts) > 2:
            raise ValueError(f"Unsupported Roman numeral: {token}")
        match = self.roman_pattern.fullmatch(parts[0])
        if not match:
            raise ValueError(f"Unsupported Roman numeral: {token}")
        numeral = match.group("roman")
        degree = DEGREES[numeral.upper()]
        accidental = match.group("acc").count("#") - match.group("acc").count("b")
        quality_mark = match.group("quality") or ""
        quality = "major" if numeral.isupper() else "minor"
        if quality_mark in {"dim", "°", "ø"}:
            quality = "diminished" if quality_mark != "ø" else "half_diminished"
        elif quality_mark in {"aug", "+"}:
            quality = "augmented"
        elif quality_mark in {"m", "min"}:
            quality = "minor"
        elif quality_mark == "maj":
            quality = "major"
        secondary = None
        if len(parts) == 2:
            secondary_token = re.sub(r"[^ivIV]", "", parts[1])
            if secondary_token.upper() not in DEGREES:
                raise ValueError(f"Unsupported secondary target: {token}")
            secondary = DEGREES[secondary_token.upper()]
        return ParsedRoman(clean, degree, accidental, numeral.isupper(), quality,
                           match.group("ext") or "", match.group("inv") or "", secondary)

    def is_valid_roman(self, token: str) -> bool:
        try:
            self.parse_roman(token)
            return True
        except ValueError:
            return False

    def _suffix_and_intervals(self, parsed: ParsedRoman) -> tuple[str, list[int]]:
        quality_intervals = {
            "major": [0, 4, 7], "minor": [0, 3, 7], "diminished": [0, 3, 6],
            "half_diminished": [0, 3, 6, 10], "augmented": [0, 4, 8],
        }
        suffixes = {"major": "", "minor": "m", "diminished": "dim", "half_diminished": "m7b5", "augmented": "aug"}
        notes = quality_intervals[parsed.quality].copy()
        suffix = suffixes[parsed.quality]
        ext = parsed.extension
        if ext == "7":
            notes.append(10 if parsed.quality != "diminished" else 9)
            suffix = "7" if parsed.quality == "major" else suffix + "7"
        elif ext == "maj7":
            notes.append(11); suffix += "maj7"
        elif ext in {"9", "11", "13"}:
            notes.extend([10, 14] + ([17] if ext in {"11", "13"} else []) + ([21] if ext == "13" else []))
            suffix += ext
        elif ext == "sus2":
            notes = [0, 2, 7]; suffix = "sus2"
        elif ext == "sus4":
            notes = [0, 5, 7]; suffix = "sus4"
        elif ext == "add9":
            notes.append(14); suffix += "add9"
        return suffix, notes

    def render_roman(self, token: str, key: str, scale: str = "major", octave: int = 4) -> RenderedChord:
        parsed = self.parse_roman(token)
        key = self.normalize_key(key)
        tonic_pc = self.pitch_class(key)
        intervals = SCALE_INTERVALS[scale]
        if parsed.secondary_degree is not None:
            target_pc = (tonic_pc + intervals[parsed.secondary_degree]) % 12
            root_pc = (target_pc + SCALE_INTERVALS["major"][parsed.degree] + parsed.accidental) % 12
        else:
            root_pc = (tonic_pc + intervals[parsed.degree] + parsed.accidental) % 12
        use_flats = "b" in key or key.upper() in FLAT_KEYS or parsed.accidental < 0
        root_name = (FLAT_NAMES if use_flats else SHARP_NAMES)[root_pc]
        suffix, chord_intervals = self._suffix_and_intervals(parsed)
        pcs = [(root_pc + interval) % 12 for interval in chord_intervals]
        base = 12 * (octave + 1) + root_pc
        midi = [base + interval for interval in chord_intervals]
        bass_name = ""
        inversion_index = {"6": 1, "64": 2, "65": 1, "43": 2, "42": 3}.get(parsed.inversion)
        if inversion_index is not None and inversion_index < len(midi):
            bass_pc = pcs[inversion_index]
            bass_name = "/" + (FLAT_NAMES if use_flats else SHARP_NAMES)[bass_pc]
            inverted = midi[inversion_index:] + [n + 12 for n in midi[:inversion_index]]
            midi = inverted
        midi = [max(36, min(96, note)) for note in midi]
        return RenderedChord(token, root_name + suffix + bass_name, midi, pcs)

    def chord_to_midi(self, chord: str, octave: int = 4) -> list[int]:
        match = re.fullmatch(r"([A-G](?:#|b)?)(m7b5|maj7|m7|dim|aug|m|7|9|11|13|sus2|sus4|add9)?(?:/([A-G](?:#|b)?))?", chord)
        if not match:
            raise ValueError(f"Unsupported chord symbol: {chord}")
        root, suffix, bass = match.groups()
        root_pc = self.pitch_class(root)
        intervals = {
            None: [0, 4, 7], "m": [0, 3, 7], "m7": [0, 3, 7, 10], "7": [0, 4, 7, 10], "maj7": [0, 4, 7, 11],
            "m7b5": [0, 3, 6, 10], "dim": [0, 3, 6], "aug": [0, 4, 8], "9": [0, 4, 7, 10, 14],
            "11": [0, 4, 7, 10, 14, 17], "13": [0, 4, 7, 10, 14, 17, 21], "sus2": [0, 2, 7],
            "sus4": [0, 5, 7], "add9": [0, 4, 7, 14],
        }[suffix]
        notes = [12 * (octave + 1) + root_pc + interval for interval in intervals]
        if bass:
            bass_note = 12 * octave + self.pitch_class(bass)
            notes = [bass_note] + notes
        return notes

    def validate_chord(self, chord: str, difficulty: str = "beginner") -> tuple[bool, str]:
        try:
            self.chord_to_midi(chord)
        except ValueError as exc:
            return False, str(exc)
        if difficulty == "beginner" and re.search(r"maj7|m7b5|dim|aug|9|11|13|sus|add|/", chord):
            return False, "Chord exceeds beginner difficulty"
        if difficulty == "intermediate" and re.search(r"m7b5|11|13", chord):
            return False, "Chord exceeds intermediate difficulty"
        return True, "valid"

    def cadence_type(self, progression: list[str]) -> str:
        if len(progression) < 2:
            return "weak"
        pair = [re.sub(r"[^ivIV]", "", token.split("/")[0]).upper() for token in progression[-2:]]
        if pair == ["V", "I"]: return "strong"
        if pair == ["IV", "I"]: return "plagal"
        if pair == ["V", "VI"]: return "deceptive"
        if pair[-1] == "V": return "half"
        return "weak"

    def voice_leading_score(self, chords: list[RenderedChord]) -> float:
        if len(chords) < 2:
            return 1.0
        movement = 0.0
        for left, right in zip(chords, chords[1:]):
            movement += sum(min(abs(a - b) for b in right.midi_notes) for a in left.midi_notes) / len(left.midi_notes)
        return round(1 / (1 + movement / (len(chords) - 1)), 4)

    def constrain_progression(self, progression: list[str], difficulty: str, cadence: str) -> tuple[list[str], list[str]]:
        reasons: list[str] = []
        valid = [token for token in progression if self.is_valid_roman(token)]
        if len(valid) != len(progression): reasons.append("Removed unsupported Roman-numeral tokens")
        if not valid:
            valid = ["I", "V", "vi", "IV"]; reasons.append("Inserted a valid baseline progression")
        simplified: list[str] = []
        for token in valid:
            replacement = token
            if difficulty == "beginner":
                replacement = re.sub(r"(maj7|add9|sus2|sus4|13|11|9|7|64|65|43|42|6)", "", replacement)
                if "/" in replacement or "°" in replacement or "ø" in replacement:
                    replacement = "V" if replacement.upper().startswith("V") else "ii"
            elif difficulty == "intermediate":
                replacement = re.sub(r"(13|11)", "7", replacement)
            if replacement != token: reasons.append(f"Simplified {token} to {replacement} for {difficulty} difficulty")
            if not simplified or not (len(simplified) >= 2 and simplified[-1] == simplified[-2] == replacement):
                simplified.append(replacement)
            else:
                reasons.append("Removed excessive chord repetition")
        endings = {"strong": ["V", "I"], "plagal": ["IV", "I"], "deceptive": ["V", "vi"], "half": ["ii", "V"]}
        ending = endings.get(cadence)
        if ending and self.cadence_type(simplified) != cadence:
            simplified = simplified[:-2] + ending if len(simplified) >= 2 else simplified + ending
            reasons.append(f"Applied {cadence} cadence")
        return simplified[:12], list(dict.fromkeys(reasons))

    def transpose_symbol(self, chord: str, semitones: int, prefer_flats: bool = False) -> str:
        match = re.fullmatch(r"([A-G](?:#|b)?)(.*?)(?:/([A-G](?:#|b)?))?", chord)
        if not match:
            raise ValueError(f"Unsupported chord symbol: {chord}")
        root, suffix, bass = match.groups()
        names = FLAT_NAMES if prefer_flats else SHARP_NAMES
        new_root = names[(self.pitch_class(root) + semitones) % 12]
        new_bass = f"/{names[(self.pitch_class(bass) + semitones) % 12]}" if bass else ""
        return new_root + suffix + new_bass

    def key_distance(self, source: str, target: str) -> int:
        return (self.pitch_class(self.normalize_key(target)) - self.pitch_class(self.normalize_key(source))) % 12


theory = TheoryEngine()
