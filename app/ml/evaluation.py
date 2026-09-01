from __future__ import annotations

import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from app.ml.common import read_jsonl, read_toml, write_json
from app.ml.parsing import parse_structured_output
from app.schemas.studio import SectionGeneration
from app.theory.engine import theory


def _baseline(row: dict[str, Any]) -> str:
    conditions = row["conditions"]
    cadence = {"Chorus": "strong", "Hook": "strong", "Bridge": "deceptive", "Outro": "plagal"}.get(conditions["section"], "half")
    base = {"strong": ["I", "IV", "V", "I"], "deceptive": ["I", "IV", "V", "vi"],
            "plagal": ["I", "bVII", "IV", "I"], "half": ["I", "vi", "ii", "V"]}[cadence]
    base, _ = theory.constrain_progression(base, conditions["difficulty"], cadence)
    return SectionGeneration(
        section_name=conditions["section"], roman_numerals=base,
        beats_per_chord=[float(int(conditions["time_signature"].split("/")[0]))] * len(base),
        cadence_type=cadence, energy_level="medium", confidence_notes="Deterministic theory baseline.",
    ).model_dump_json()


def _token_scores(predicted: list[str], expected: list[str]) -> tuple[float, float, float]:
    pred, gold = Counter(predicted), Counter(expected)
    correct = sum((pred & gold).values())
    precision = correct / max(1, sum(pred.values())); recall = correct / max(1, sum(gold.values()))
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return precision, recall, f1


def _difficulty_ok(tokens: list[str], difficulty: str) -> bool:
    if difficulty == "beginner": return not any(any(mark in token for mark in ["7", "9", "11", "13", "sus", "/", "°", "ø"]) for token in tokens)
    if difficulty == "intermediate": return not any(any(mark in token for mark in ["11", "13", "ø"]) for token in tokens)
    return True


def _aggregate(system: str, rows: list[dict[str, Any]], outputs: list[str], latencies: list[float], loss: float | None) -> dict[str, Any]:
    details = []
    parsed_progressions: list[tuple[str, ...]] = []
    for row, raw in zip(rows, outputs):
        expected = SectionGeneration.model_validate_json(row["response"])
        parsed, repaired_parse, _ = parse_structured_output(raw)
        if parsed:
            valid_romans = [theory.is_valid_roman(token) for token in parsed.roman_numerals]
            chord_valid = []
            for token in parsed.roman_numerals:
                try:
                    theory.render_roman(token, row["conditions"]["key"], row["conditions"]["scale"]); chord_valid.append(True)
                except Exception: chord_valid.append(False)
            constrained, reasons = theory.constrain_progression(parsed.roman_numerals, row["conditions"]["difficulty"], expected.cadence_type)
            precision, recall, f1 = _token_scores(parsed.roman_numerals, expected.roman_numerals)
            progression = tuple(parsed.roman_numerals); parsed_progressions.append(progression)
            details.append({
                "parse": 1, "precision": precision, "recall": recall, "f1": f1,
                "exact": int(parsed.roman_numerals == expected.roman_numerals),
                "roman_validity": sum(valid_romans) / max(1, len(valid_romans)),
                "chord_validity": sum(chord_valid) / max(1, len(chord_valid)),
                "cadence": int(theory.cadence_type(parsed.roman_numerals) == expected.cadence_type),
                "difficulty": int(_difficulty_ok(parsed.roman_numerals, row["conditions"]["difficulty"])),
                "repair": int(bool(reasons) or repaired_parse or constrained != parsed.roman_numerals),
            })
        else:
            details.append({key: 0 for key in ["parse", "precision", "recall", "f1", "exact", "roman_validity", "chord_validity", "cadence", "difficulty", "repair"]})
    def average(key: str) -> float:
        return round(sum(item[key] for item in details) / max(1, len(details)), 6)
    diversity = len(set(parsed_progressions)) / max(1, len(parsed_progressions))
    return {
        "system": system, "examples": len(rows), "validation_loss": None if loss is None else round(loss, 6),
        "perplexity": None if loss is None else round(math.exp(min(loss, 20)), 6),
        "token_precision": average("precision"), "token_recall": average("recall"), "token_f1": average("f1"),
        "exact_progression_match": average("exact"), "json_parse_success_rate": average("parse"),
        "roman_numeral_validity_rate": average("roman_validity"), "chord_validity_rate": average("chord_validity"),
        "cadence_satisfaction_rate": average("cadence"), "difficulty_compliance_rate": average("difficulty"),
        "progression_diversity": round(diversity, 6), "duplicate_rate": round(1 - diversity, 6),
        "constraint_repair_rate": average("repair"), "average_inference_latency_ms": round(statistics.mean(latencies), 3) if latencies else 0.0,
    }


def _neural_outputs(model, tokenizer, rows: list[dict[str, Any]], seed: int) -> tuple[list[str], list[float], float]:
    import torch
    outputs, latencies, losses = [], [], []
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        for row in rows:
            torch.manual_seed(seed)
            encoded = tokenizer(row["instruction"], return_tensors="pt", truncation=True, max_length=512).to(device)
            labels = tokenizer(text_target=row["response"], return_tensors="pt", truncation=True, max_length=256).input_ids.to(device)
            losses.append(float(model(**encoded, labels=labels).loss.item()))
            started = time.perf_counter()
            generated = model.generate(**encoded, max_new_tokens=256, do_sample=False)
            latencies.append((time.perf_counter() - started) * 1000)
            outputs.append(tokenizer.decode(generated[0], skip_special_tokens=True))
    return outputs, latencies, statistics.mean(losses)


def evaluate_systems(config_path: str | Path, limit: int | None = None) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    config = read_toml(config_path); seed = int(config["seed"])
    rows = read_jsonl(Path(config["dataset_dir"]) / "test.jsonl")
    if limit: rows = rows[:limit]
    if not rows: raise RuntimeError("Evaluation test split is empty")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    base = AutoModelForSeq2SeqLM.from_pretrained(config["base_model"]).to(device)
    base_outputs, base_latencies, base_loss = _neural_outputs(base, tokenizer, rows, seed)
    base_metrics = _aggregate("unmodified_flan_t5_small", rows, base_outputs, base_latencies, base_loss)
    del base
    adapted_base = AutoModelForSeq2SeqLM.from_pretrained(config["base_model"])
    adapted = PeftModel.from_pretrained(adapted_base, config["output_dir"]).to(device)
    lora_outputs, lora_latencies, lora_loss = _neural_outputs(adapted, tokenizer, rows, seed)
    lora_metrics = _aggregate("flan_t5_small_lora", rows, lora_outputs, lora_latencies, lora_loss)
    baseline_started = time.perf_counter(); baseline_outputs = [_baseline(row) for row in rows]
    baseline_elapsed = (time.perf_counter() - baseline_started) * 1000 / len(rows)
    baseline_metrics = _aggregate("deterministic_theory_baseline", rows, baseline_outputs, [baseline_elapsed] * len(rows), None)
    comparison = {
        metric: round(lora_metrics[metric] - base_metrics[metric], 6)
        for metric in ["token_f1", "exact_progression_match", "json_parse_success_rate", "roman_numeral_validity_rate",
                       "chord_validity_rate", "cadence_satisfaction_rate", "difficulty_compliance_rate"]
    }
    configured_steps = int(config.get("max_steps", 0))
    training_scope = (
        f"The development adapter ran for {configured_steps} optimizer steps on a small synthetic dataset. "
        "Loss improvement does not establish production-quality structured generation."
    )
    report = {
        "base_model": config["base_model"], "adapter_path": config["output_dir"], "device": device,
        "examples": len(rows), "systems": [base_metrics, lora_metrics, baseline_metrics],
        "lora_minus_base": comparison,
        "summary": {"base_token_f1": base_metrics["token_f1"], "lora_token_f1": lora_metrics["token_f1"],
                    "baseline_token_f1": baseline_metrics["token_f1"], "development_run_limitation": training_scope},
    }
    destination = Path("storage/evaluation_results.json"); write_json(destination, report)
    labels = [item["system"] for item in report["systems"]]
    metrics = [
        "validation_loss", "perplexity", "token_precision", "token_recall", "token_f1", "exact_progression_match",
        "json_parse_success_rate", "roman_numeral_validity_rate", "chord_validity_rate", "cadence_satisfaction_rate",
        "difficulty_compliance_rate", "progression_diversity", "duplicate_rate", "constraint_repair_rate", "average_inference_latency_ms",
    ]
    lines = ["# Actual Development Evaluation", "", f"Evaluated {len(rows)} held-out examples on {device}.", "",
             f"| Metric | {labels[0]} | {labels[1]} | {labels[2]} |", "|---|---:|---:|---:|"]
    for metric in metrics:
        values = [item[metric] if item[metric] is not None else "n/a" for item in report["systems"]]
        lines.append(f"| {metric} | {values[0]} | {values[1]} | {values[2]} |")
    lines.extend(["", training_scope, "Metrics above are calculated, not estimated."])
    Path("EVALUATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
