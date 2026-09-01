# Model Card: JDG FLAN-T5-small Lyric-to-Chord LoRA

## Model

- Base: `google/flan-t5-small`
- Task: instruction-to-JSON Roman-numeral chord progression generation
- Method: PEFT LoRA on T5 query and value attention projections
- Default adapter: rank 8, alpha 16, dropout 0.05, no bias
- Formats: separate base-model cache plus Safetensors adapter
- Required inference device: CPU; CUDA is optional

## Training data

All instructions and targets are procedural and copyright-safe. Lyrics are short original lines authored for this repository. Targets come from a deterministic music-theory generator. Development data proves the path affordably; `dataset_full.toml` produces at least 30,000 examples across the documented musical dimensions.

## Intended use

This model is a creative assistant for original-song chord ideation, education, and local portfolio demonstration. It proposes key-independent Roman numerals that are always checked and rendered by deterministic theory code. It is not intended to reproduce the harmony of copyrighted songs, identify songs, or provide an authoritative musicological analysis.

## Reproducibility

Dataset, training, and decoding seeds are explicit. Training settings reside in TOML. The trainer confirms target-module names, reports total/trainable parameters, rejects non-LoRA trainables, saves Safetensors, and writes `training_report.json`. Evaluation compares the unmodified base, adapter, and theory baseline on the held-out test split.

## Evaluation

The executed 80-step CPU run trained 344,064 of 77,305,216 parameters (0.445072%) and saved `adapter_model.safetensors`. On nine held-out examples, base validation loss/perplexity were 3.266581/26.221539 and LoRA loss/perplexity were 1.437104/4.208492. The adapter moved from empty output to non-empty learned text, but both neural decoders had 0% strict JSON parse success, so no structured output-quality improvement is claimed. The theory baseline produced 100% parse, Roman, and chord validity with token F1 0.521164. Complete calculated values are in `storage/evaluation_results.json` and `EVALUATION_REPORT.md`.

## Limitations and expected failures

- This 80-step adapter learns the target token distribution but does not yet reliably close the structured JSON contract.
- FLAN-T5-small may truncate, omit fields, hallucinate unsupported notation, or ignore nuanced lyric emotion.
- Sampling is seed-reproducible on the same software/device stack but exact cross-platform floating-point reproducibility is not guaranteed.
- The English syllable and emotion heuristics do not generalize equally to every language.
- The deterministic constraint layer can alter cadence or simplify advanced tokens; the UI discloses every repair.
- If either base or adapter is missing, the application may use the explicitly labeled deterministic fallback.

## Privacy and licensing

Inference is local. Full lyrics are not logged. Users are responsible for submitting text they are allowed to use. The repository is MIT licensed; the base model retains its upstream license and model-card terms.
