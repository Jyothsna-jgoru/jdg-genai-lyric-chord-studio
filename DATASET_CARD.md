# Dataset Card: JDG Synthetic Lyric-to-Chord Instructions

## Summary

This repository generates instruction/response examples for key-independent Roman-numeral chord progression generation. No copyrighted lyric or chord site is scraped. Every lyric is selected from a small collection of original repository-authored lines, and every target is produced by deterministic theory rules with seeded diversity.

## Configurations

- Development: 96 records, seed 42, 75%/12.5%/12.5% target split ratios.
- Full: 30,000 records, seed 42, 80%/10%/10% target split ratios.

The executed development run produced 73 training, 14 validation, and 9 test records. Validation found zero cross-split group leakage, zero schema/theory errors, and a 0.010417 duplicate-response rate. Actual split counts vary slightly because a stable hash assigns condition/progression groups rather than forcing individual record counts. This prevents identical musical conditions with identical targets from crossing splits.

## Coverage

The generator covers all 12 chromatic key choices; major, natural minor, harmonic minor, and melodic minor scales; Intro, Verse, Pre-Chorus, Chorus, Hook, Bridge, Breakdown, and Outro; pop, rock, acoustic, ballad, folk, blues, R&B, and cinematic genres; eight moods; three difficulties; 50–180 BPM; five meters; four densities; five cadence families; previous-section context; repeated-line flags; and controlled sevenths, suspensions, inversions, borrowed chords, secondary dominants, diminished passing chords, and extensions.

## Schema

Each JSONL object has a unique ID, complete delimited instruction, serialized strict-JSON response, structured conditions, and a hash group key. Responses contain `section_name`, `roman_numerals`, `beats_per_chord`, `cadence_type`, `energy_level`, and `confidence_notes`.

## Quality controls

`dataset-validate` reparses every target with Pydantic, validates every Roman token through the independent theory engine, recalculates distributions and duplicate response rate, and fails if a group appears across splits. `statistics.json` and `validation.json` contain actual run results.

## Limitations

The lyric vocabulary is deliberately limited and synthetic, so the data is safe but less linguistically diverse than real songs. The target generator encodes particular Western tonal-harmony assumptions. Duplicate response strings are expected because many different lyric/control inputs can validly share a common progression; group leakage, rather than response-string uniqueness alone, is the split safety criterion.
