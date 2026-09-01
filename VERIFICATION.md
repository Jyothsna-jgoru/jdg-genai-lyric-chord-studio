# Verification Record

Verification date: 2026-09-01. Commands ran from the repository root with the Python 3.12 virtual environment.

## Environment and installation

Command:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Result: successful editable install. Verified versions included Python 3.12.13, FastAPI 0.141.1, PyTorch 2.13.0+cpu, Transformers 4.57.6, and PEFT 0.20.0.

## Dataset

Commands:

```powershell
.venv\Scripts\python.exe -m app.cli dataset-generate --config configs/dataset_dev.toml
.venv\Scripts\python.exe -m app.cli dataset-validate --dataset-dir data/generated/dev
```

Actual result: 96 examples; 73 train, 14 validation, 9 test; zero cross-split group leakage; duplicate response rate 0.010417; zero validation errors; `valid: true`.

## LoRA training and checkpoint

Command:

```powershell
.venv\Scripts\python.exe -m app.cli train --config configs/train_dev.toml
```

Actual result: loaded cached `google/flan-t5-small` fully offline, confirmed T5 target modules `q` and `v`, and trained for 80 optimizer steps (1.0959 epochs) on CPU. Total parameters were 77,305,216; trainable LoRA parameters were 344,064 (0.445072%); the trainable-isolation assertion passed. Mean training loss was 2.466541 and runtime was 327.4346 seconds. Final trainer validation loss was 1.396888. PEFT saved `storage/adapters/dev/adapter_model.safetensors` and the adapter reloaded successfully.

## Base and adapter inference

Commands:

```powershell
.venv\Scripts\python.exe -m app.cli infer --base-only --lyrics "Morning opens every window"
.venv\Scripts\python.exe -m app.cli infer --lyrics "Morning opens every window"
```

Actual result: base and LoRA models both loaded on CPU and executed offline. In a representative request, the base decoder returned an empty string in 820.896 ms; the LoRA decoder returned non-empty learned schema-like text in 2960.152 ms, but it was not valid JSON. The application correctly rejected malformed structured output and identified its usable response as `algorithmic_fallback_output` rather than presenting it as model output.

## Comparative evaluation

Command:

```powershell
.venv\Scripts\python.exe -m app.cli evaluate --config configs/train_dev.toml
```

Actual result: nine held-out examples evaluated for the unmodified base, 80-step LoRA adapter, and deterministic theory baseline. Base loss/perplexity were 3.266581/26.221539; LoRA loss/perplexity were 1.437104/4.208492. Both neural systems had zero strict JSON parse success and zero token F1. The baseline had JSON/Roman/chord validity 1.0, difficulty compliance 1.0, token F1 0.521164, exact match 0.111111, and mean latency 0.142 ms. Full metrics are saved in `storage/evaluation_results.json` and `EVALUATION_REPORT.md`.

## Automated tests

Command:

```powershell
.venv\Scripts\python.exe -m pytest
```

Final result after fixes: `20 passed, 2 warnings in 9.77s`. The model-marked test loaded the actual adapter and ran inference. Tests used project-local temporary directories and temporary SQLite databases; they did not mutate the normal test database fixture.

Covered behavior included deterministic dataset generation and split leakage, schema parsing and repair, model/adapter absence, trained adapter artifacts and inference, lyric headings/syllables/emotion, scale spelling, Roman conversion, borrowed/secondary harmony, cadence constraints, major/minor-seventh MIDI realization, MIDI voicing, voice leading, transposition, deterministic fallback, control-sensitive musical fingerprints, chord-density alignment, lyric-shaped playback melodies, different melody fingerprints for different lyrics, the accessible generation-progress contract, the three-track MIDI structure, timeline, all four exports, input limits, invalid tempo, empty lyrics, project CRUD, duplicate/delete confirmation, health, and API errors.

## Server and Docker configuration

Application command:

```powershell
.venv\Scripts\python.exe -m app.cli serve --host 127.0.0.1 --port 8000
```

Actual result: startup loaded the base and `dev` adapter once on CPU; `GET /` returned HTTP 200; the application ran at `http://127.0.0.1:8000/`.

Docker validation command:

```powershell
docker compose config --quiet
```

Actual result: Compose configuration valid with Docker 29.7.2 and Compose 5.4.0. A container image build was not run because native installation, training, inference, server startup, and testing were the selected verified execution path.

## Browser-verified workflows

The refreshed professional interface was exercised in the local in-app browser using semantic Playwright controls:

- Loaded the responsive navy, teal, white, and blue-gray interface with no console errors.
- Analyzed the bundled lyrics: two sections and four lyric lines detected.
- Generated two chord-sheet sections. The UI honestly displayed `algorithmic_fallback_output` because the 80-step adapter did not emit parseable JSON.
- Verified control sensitivity at a fixed key: hopeful/seed 42 produced `C Am C Dm Am F C Am`; dark/seed 42 produced `C F F Dm C F Am Dm`; dark/seed 43 produced `Am F C Dm C F F Dm`.
- Regenerated only the Verse and confirmed the UI advanced the seed from 43 to 44, reported `Verse regenerated with seed 44`, and changed the arrangement while preserving both sections.
- Regenerated the Verse while preserving two visible song sections.
- Changed the first manual chord to F, transposed from C to D, and confirmed the current manual chord became G.
- Saved, reopened, renamed, and duplicated the project; the project selector displayed both saved records.
- Started playback, paused with a stable `(paused)` indicator, resumed, restarted, and stopped. A race condition found during this check was fixed with playback cancellation tokens and retested.
- Triggered TXT, JSON, PDF, and MIDI downloads; each control reported successful completion. Automated integration tests independently asserted TXT content, JSON structure, PDF signature, and MIDI header bytes.
- Confirmed zero browser console errors after the final run.

Project deletion was verified by the API integration test, including the required rejection without `confirm=true` and successful deletion with confirmation. The browser-created demonstration projects were intentionally retained for the open app preview.

## Remaining limitations

- The affordable 80-step development adapter proves real LoRA mechanics and substantially lowers loss, but is not sufficiently trained to emit reliable JSON. Full 30,000-example training is configured but was not executed.
- Exact seeded output can vary across different PyTorch/device versions even though same-environment requests are deterministic.
- The local English syllable and emotion heuristics and the Western tonal theory vocabulary are deliberately bounded.
- Docker configuration was validated but the image itself was not built in this run.
