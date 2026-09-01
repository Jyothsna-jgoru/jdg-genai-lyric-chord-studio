# JDG GenAI Lyric-to-Chord Studio

Built by Jyothsna Devi Goru.

JDG GenAI Lyric-to-Chord Studio is a local, portfolio-ready Generative AI application that turns original lyrics into editable, playable, and exportable chord sheets. Its neural path uses `google/flan-t5-small` with a PEFT LoRA adapter trained on procedurally generated, copyright-safe instruction data. Every neural result passes through strict JSON validation and an independent music-theory constraint engine. When the model or adapter is absent, the application labels its deterministic theory fallback honestly.

No API key, subscription, billing account, lyric scraping, or hosted inference service is used. After dependencies and the public base model are cached, the application can run offline.

## Working features

- Local lyric-section detection and editable section boundaries.
- Line-level word, syllable, repetition, punctuation, position, emotional-score, chord-change, and rhythmic-density features.
- FLAN-T5-small sequence-to-sequence inference with optional LoRA adapter loading once at startup.
- Greedy, beam, temperature, and top-k decoding with deterministic seeds.
- Pydantic JSON validation, one deterministic parse repair, theory repair, and explicit generation-source provenance.
- Major, natural minor, harmonic minor, and melodic minor scales; Roman numerals; borrowed harmony; secondary dominants; inversions; suspensions; extensions; cadence checks; voice-leading scoring; MIDI notes; and enharmonic spelling.
- SQLite project create, list, open, update, rename, duplicate, and confirmed delete.
- Editable chord positions above lyric words, transposition without regeneration, and Web Audio playback with four local timbres.
- Deterministic lyric-shaped lead melodies derived from syllables, emotional features, key, scale, mood, variation, and seed, layered above the generated chord accompaniment.
- An accessible generation progress panel with a rotating loader, elapsed time, realistic one-minute-to-90-second CPU guidance, staged status messages, and duplicate-request prevention.
- TXT, structured JSON, PDF, and three-track Standard MIDI exports that use current manual edits and transposition; MIDI contains metadata, accompaniment, and lyric-shaped melody tracks.
- Reproducible development and 30,000-example dataset configurations, genuine PEFT training, checkpoint resume, adapter merge, three-system evaluation, optional local MLflow tracking, Docker, and automated tests.

## Why FLAN-T5-small

FLAN-T5-small is an instruction-tuned encoder-decoder model whose size is practical for a CPU development run. The encoder consumes explicitly separated musical controls and inert lyric data, while the decoder learns the compact JSON chord schema. Roman numerals keep the neural task key-independent; deterministic theory code performs key-specific spelling and MIDI realization.

## GenAI architecture

```text
original lyrics + controls
        │
        ├─ local lyric analysis ──► delimited instruction
        │                                  │
        │                        FLAN-T5-small + LoRA(q,v)
        │                                  │
        │                           generated JSON text
        │                                  │
        └────────────────────► Pydantic parse + one repair
                                           │
                                  theory constraints
                                           │
                         chords + alignment + provenance
                           │         │          │
                       Web Audio   SQLite   TXT/JSON/PDF/MIDI
```

LoRA rank, alpha, dropout, target modules, lengths, optimizer settings, accumulation, seed, precision, and early stopping are configured in `configs/train_dev.toml` and `configs/train_full.toml`. Training inspects the loaded T5 module names before accepting `q` and `v`, freezes non-adapter weights, prints total/trainable counts, and saves only `adapter_model.safetensors` plus adapter configuration. The executed run contained 77,305,216 total parameters and 344,064 trainable LoRA parameters (0.445072%); `storage/adapters/dev/training_report.json` is the authoritative record.

## Native setup

Python 3.12 is required. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The first model operation downloads the public FLAN-T5-small files. Later runs use the Hugging Face cache and can be offline.

## Exact commands

Generate and validate the affordable development data:

```bash
python -m app.cli dataset-generate --config configs/dataset_dev.toml
python -m app.cli dataset-validate --dataset-dir data/generated/dev
```

Generate the full 30,000-example dataset:

```bash
python -m app.cli dataset-generate --config configs/dataset_full.toml
python -m app.cli dataset-validate --dataset-dir data/generated/full
```

Run the short real LoRA proof or full training:

```bash
python -m app.cli train --config configs/train_dev.toml
python -m app.cli train --config configs/train_full.toml
```

Resume a saved Trainer checkpoint:

```bash
python -m app.cli resume --config configs/train_dev.toml --checkpoint checkpoints/dev/checkpoint-1
```

Evaluate the unmodified base model, LoRA model, and deterministic theory baseline:

```bash
python -m app.cli evaluate --config configs/train_dev.toml
```

Run one base or adapter inference and optionally merge the adapter:

```bash
python -m app.cli infer --base-only --lyrics "Morning opens every window"
python -m app.cli infer --lyrics "Morning opens every window"
python -m app.cli merge --config configs/train_dev.toml --destination storage/merged-model
```

Start the complete application and open `http://127.0.0.1:8000`:

```bash
python -m app.cli serve
```

Run every automated test:

```bash
python -m pytest
```

## Docker

```bash
docker compose build
docker compose up
```

The Compose service persists the database, adapters, model cache, and exports in named volumes and exposes port 8000. Its first start needs internet access for the public base model unless the cache volume is already populated. The service runs as a non-root user and has an HTTP health check.

## API examples

Analyze lyrics:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/lyrics/analyze -H "Content-Type: application/json" -d '{"lyrics":"[Verse]\nMorning opens every window"}'
```

Generate chords:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chords/generate -H "Content-Type: application/json" -d '{"project_name":"Morning Song","lyrics":"[Verse]\nMorning opens every window","controls":{"key":"C","scale":"major","genre":"pop","mood":"hopeful","tempo":96,"time_signature":"4/4","difficulty":"beginner","chord_density":2,"variation":0.35,"seed":42,"decoding_method":"greedy","num_beams":4,"temperature":0.8,"top_k":40}}'
```

Interactive API documentation is available at `/docs` while the server runs.

## Dataset generation

The generator combines scales, keys, sections, genres, moods, tempi, meters, difficulty, density, cadence, repetition, and controlled advanced harmony. Targets are produced by deterministic theory code. A stable hash of identical musical conditions and progression assigns every equivalent group to one split, preventing group leakage. Generated JSONL is intentionally ignored by Git; the repository keeps configs, code, statistics produced by a run, and a short original lyric sample.

## Local MLflow

Set `mlflow = true` in a training TOML configuration. Training writes to the local file store under `storage/mlruns`; no server or account is required. Parameters, Trainer metrics, dataset/model configuration, and adapter artifacts are retained locally.

## Actual development evaluation

The executed nine-example CPU evaluation measured validation loss 3.266581 for the base model and 1.437104 for the 80-step LoRA adapter, reducing perplexity from 26.221539 to 4.208492. The adapter now emits learned text but still has 0% strict JSON parse success, while the deterministic theory baseline achieved 100% JSON/Roman/chord validity and token F1 0.521164. The application therefore rejects malformed neural output and uses its explicitly labeled theory fallback. The fallback is input-sensitive: mood, genre, tempo, density, variation, lyrics, decoding method, and seed all affect its musical fingerprint. No structured neural quality gain is claimed; see `EVALUATION_REPORT.md` for every calculated metric.

## Privacy and safety

Lyrics remain on the machine and are never sent to an external inference API. Application logs include request IDs but omit complete lyrics. Lyrics are delimited as inert data in every model instruction, rendered with text-only DOM APIs, length-limited, and never executed as code or HTML. Project and download names are sanitized. SQLite writes use scoped SQLAlchemy sessions and transactions.

## Repository structure

```text
app/api/          versioned FastAPI routes
app/core/         settings, request IDs, structured logging
app/db/           SQLAlchemy engine and project model
app/ml/           dataset, parsing, model loading, LoRA training, evaluation
app/schemas/      Pydantic requests, responses, and model-output schema
app/services/     lyric, generation, persistence, and export services
app/static/       plain HTML, CSS, and vanilla JavaScript interface
app/theory/       independent music-theory constraint engine
configs/          development and full data/training configurations
data/samples/     short repository-authored sample lyrics
storage/          runtime database, adapters, evaluations, MLflow, exports
tests/            unit, integration, API, artifact, and export tests
```

## Troubleshooting

- If health reports an unavailable model, run an inference or training command once with internet access so FLAN-T5-small enters the local cache.
- If the adapter is unavailable, run the development training command. The UI will continue with an explicitly labeled algorithmic fallback.
- If CPU training is slow, keep the 80-step development config for verification and reserve `train_full.toml` for a longer run or CUDA machine.
- If a manual chord blocks playback/export, correct it to a supported spelling such as `C`, `F#m`, `Bbmaj7`, `G7/B`, or `Dsus4`.
- Major, minor, dominant-seventh, major-seventh, minor-seventh (for example `Ebm7`), suspended, extended, diminished, augmented, and supported slash chords are realized consistently for playback and MIDI export.
- After changing lyrics or musical controls, click **Generate complete song** again. The interface marks the old chord sheet as stale and blocks playback, saving, transposition, and export until a fresh arrangement is generated.
- Browser audio begins only after a button press because browsers require a user gesture before creating sound.

## Known limitations

The 80-step development adapter demonstrates genuine gradient updates, Safetensors checkpointing, reload, non-empty learned inference, and lower validation loss; it still does not produce reliable structured JSON. Full training is reproducible but intentionally not run as part of the affordable verification. Syllables, emotion, and the lead melody are deterministic English-oriented heuristics. The generated melody is an original sketch for the submitted lyrics, not a reconstruction of an existing recording. Advanced Roman-numeral support is broad but not a complete implementation of every jazz or microtonal notation. Web Audio timbres are lightweight synthesis, not sampled acoustic instruments.

See [TECHNICAL_DESIGN.md](TECHNICAL_DESIGN.md), [MODEL_CARD.md](MODEL_CARD.md), [DATASET_CARD.md](DATASET_CARD.md), [EVALUATION_REPORT.md](EVALUATION_REPORT.md), and [VERIFICATION.md](VERIFICATION.md).
