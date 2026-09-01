# Technical Design

## System boundary

The application is a FastAPI process serving a plain static interface and versioned JSON endpoints. It owns a SQLite file, a cached FLAN-T5 base model, a separate PEFT adapter directory, and transient export bytes. No inference data crosses the process boundary.

## Instruction format and sequence-to-sequence flow

`app.ml.dataset.build_instruction` is shared by training and inference. Every record explicitly names Task, Lyrics, detected section, key, scale, genre, mood, tempo, time signature, difficulty, chord density, variation, previous progression, lyric features, and the requested JSON schema. Lyrics appear only between `<LYRICS_DATA>` delimiters after a statement that they are inert user data. The T5 encoder maps this instruction into hidden states; the decoder autoregressively produces the compact `SectionGeneration` JSON document.

## LoRA target modules and adapter training

Training loads `google/flan-t5-small`, enumerates `named_modules`, and requires both configured suffixes `.q` and `.v` to exist before PEFT wrapping. `LoraConfig` uses `SEQ_2_SEQ_LM`, rank 8, alpha 16, dropout 0.05, no bias, and only q/v targets by default. A hard assertion rejects any trainable parameter whose name lacks `lora_`. AdamW, linear warmup/decay, gradient accumulation, seed, lengths, mixed precision, checkpoints, and early stopping are controlled in TOML. Validation includes generated-output chord validity. The early-stopping callback resets patience when either validation loss or chord-validity improves and stops only when neither improves; best-checkpoint restoration uses validation loss. Base weights remain in the Hugging Face cache and the adapter is saved as Safetensors.

## Constrained generation and provenance

The model service is a process singleton guarded by an inference lock. It selects CPU by default and CUDA when available, loads once, and supports greedy, beam, temperature, and top-k decoding. Torch seeds are set before every request.

Generated text is first validated directly as `SectionGeneration`. On failure, exactly one deterministic repair extracts the outer JSON object, removes trailing commas, and converts a wholly single-quoted Python literal. A second failure stops the neural path. Parsed Roman tokens then enter `TheoryEngine.constrain_progression`, which removes unsupported tokens, simplifies by difficulty, limits triple repetition, and applies the requested cadence while retaining unaffected tokens. Raw text, the original token list, repaired list, and repair reasons remain in the response and project record. Provenance is one of `lora_model_output`, `lora_output_repaired_by_constraints`, `base_model_output`, or `algorithmic_fallback_output` (or an explicit combination across sections).

## Theory engine

The engine maps pitch names to pitch classes, spells heptatonic scales letter-by-letter, parses accidentals and Roman degrees, realizes primary and secondary functions, attaches quality/extensions, handles inversions, emits chord symbols and MIDI voicings, validates difficulty, detects cadences, estimates voice-leading movement, and transposes symbols chromatically. It is imported by services, datasets, evaluation, and tests; no theory rules live in API handlers.

## Lyric analysis and alignment

Bracketed, parenthesized, colon-terminated, and plain section labels are recognized. Missing labels produce one Verse. Each non-heading line receives counts and deterministic heuristic features using a bundled English emotion lexicon. Alignment selects no more chord positions than the line's words, spreads changes across word indexes, and cycles through the validated progression. The interface exposes every word position so a user can clear, add, or change a chord. Manual values are validated before audio and exports.

## Audio timeline

The backend returns ordered events containing section, lyric line and fragment, chord MIDI pitches, lyric-shaped melody pitches, starting beat, and duration. A stable SHA-256 motif seed combines the complete submitted lyrics with the local fragment, syllable count, emotional score, key, scale, chord, genre, mood, tempo, variation, and user seed. Scale-constrained contour rules then create an original monophonic lead line whose rhythmic subdivision follows the fragment's estimated syllables. The browser converts beats using the current tempo and schedules the lead line above quieter chord oscillators through separate envelope/filter choices and a master gain. Piano-like, pad, pluck, and clean-synth choices change oscillator/envelope characteristics. An AudioContext clock drives the now-playing lyric-fragment display; suspended contexts stop advancing. Starting or restarting first stops every previous source, preventing overlapping sessions. Editing lyrics or controls marks the current arrangement stale and blocks downstream use until regeneration, preventing playback of an earlier song under new inputs.

## Database design

One `projects` table stores project identity, original lyrics, detected and edited sections, controls, analysis, raw model text, validated output, repairs, alignment, manual edits, seed, decoding settings, provenance, model/adapter versions, and UTC timestamps. Unique names prevent accidental overwrite. Per-request SQLAlchemy sessions commit atomically and roll back conflicts. Delete requires both a UI confirmation and `confirm=true` at the endpoint.

## Export pipeline

All exports consume the current validated response, including its key and manual alignment edits. TXT produces a monospaced chord sheet. JSON preserves the full typed response. ReportLab creates a readable PDF without rendering lyrics as HTML. Mido creates a Type-1 MIDI file with separate metadata, chord-accompaniment, and lyric-shaped melody tracks, plus tempo, time signature, section markers, note-on/note-off events, and beat-derived durations. Transposition mutates symbols before playback and export, so every downstream format remains consistent.

## Error and security behavior

Pydantic limits lyrics, control ranges, meters, and project names. A request-size middleware rejects oversized bodies. Central handlers produce stable JSON errors, request IDs flow through response headers and logs, and complete lyrics are not logged. The frontend inserts all user data with `textContent`/`value`; it never uses `innerHTML` or evaluates user text.
