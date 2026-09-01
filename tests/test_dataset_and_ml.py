from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.ml.dataset import generate_dataset, validate_dataset
from app.ml.model import ModelService
from app.ml.parsing import parse_structured_output
from app.schemas.studio import MusicalControls


def test_dataset_reproducibility_and_leakage(tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    configs = []
    for output in (first, second):
        config = tmp_path / f"{output.name}.toml"
        config.write_text(f'size = 80\nseed = 7\noutput_dir = "{output.as_posix()}"\ntrain_ratio = 0.7\nvalidation_ratio = 0.15\n', encoding="utf-8")
        configs.append(config)
    stats_one = generate_dataset(configs[0]); stats_two = generate_dataset(configs[1])
    assert stats_one["split_sizes"] == stats_two["split_sizes"]
    assert (first / "train.jsonl").read_bytes() == (second / "train.jsonl").read_bytes()
    validated = validate_dataset(first)
    assert validated["valid"] is True
    assert validated["cross_split_group_leakage"] == 0


def test_structured_output_and_single_repair():
    valid = '{"section_name":"Verse","roman_numerals":["I","V"],"beats_per_chord":[4,4],"cadence_type":"half","energy_level":"medium","confidence_notes":"ok"}'
    parsed, repaired, _ = parse_structured_output(valid); assert parsed and not repaired
    parsed, repaired, _ = parse_structured_output("prefix " + valid[:-1] + ",} suffix")
    assert parsed and repaired
    parsed, repaired, error = parse_structured_output("not json"); assert parsed is None and repaired and error


def test_model_and_adapter_absence_fails_gracefully(monkeypatch, tmp_path):
    import app.ml.model as model_module
    fake = SimpleNamespace(base_model=str(tmp_path / "missing-model"), allow_model_download=False,
                           adapter_path=tmp_path / "missing-adapter", evaluation_path=tmp_path / "missing.json")
    monkeypatch.setattr(model_module, "settings", fake)
    service = ModelService(); service.load(allow_download=False)
    assert service.base_loaded is False
    output, latency = service.generate("instruction", MusicalControls())
    assert output is None and latency == 0
    assert service.health()["fallback_engine_available"] is True


@pytest.mark.model
def test_trained_adapter_artifacts_and_inference():
    root = Path(__file__).resolve().parents[1]
    adapter = root / "storage" / "adapters" / "dev"
    if not (adapter / "adapter_model.safetensors").exists():
        pytest.skip("Development adapter has not been trained in this checkout")
    report = json.loads((adapter / "training_report.json").read_text(encoding="utf-8"))
    assert report["only_lora_trainable"] is True
    assert report["trainable_parameters"] < report["total_parameters"]
    service = ModelService(); service.load(allow_download=False)
    assert service.base_loaded and service.adapter_loaded
    output, latency = service.generate("Task: Return JSON. Lyrics: <LYRICS_DATA>Morning light</LYRICS_DATA>", MusicalControls())
    assert isinstance(output, str) and latency > 0

