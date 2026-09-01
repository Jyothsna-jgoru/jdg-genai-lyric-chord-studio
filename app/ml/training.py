from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

from app.ml.common import read_toml, write_json
from app.ml.parsing import parse_structured_output
from app.theory.engine import theory


def train_adapter(config_path: str | Path, resume_from: str | None = None) -> dict[str, Any]:
    import numpy as np
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
        Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback,
    )

    config = read_toml(config_path)
    seed = int(config["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model_name = config["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    module_names = [name for name, _ in base_model.named_modules()]
    requested_targets = list(config["target_modules"])
    confirmed_targets = [target for target in requested_targets if any(name.endswith(f".{target}") for name in module_names)]
    if confirmed_targets != requested_targets:
        raise RuntimeError(f"T5 target module confirmation failed: requested={requested_targets} confirmed={confirmed_targets}")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, r=int(config["lora_rank"]), lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]), target_modules=confirmed_targets, bias="none",
    )
    model = get_peft_model(base_model, peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    illegal_trainables = [name for name, p in model.named_parameters() if p.requires_grad and "lora_" not in name]
    if illegal_trainables:
        raise RuntimeError(f"Non-LoRA parameters are trainable: {illegal_trainables[:10]}")
    parameter_report = {
        "total_parameters": total, "trainable_parameters": trainable,
        "trainable_percentage": round(100 * trainable / total, 6),
        "confirmed_target_modules": confirmed_targets, "only_lora_trainable": True,
    }
    print(json.dumps(parameter_report, indent=2))

    dataset_dir = Path(config["dataset_dir"])
    dataset = load_dataset("json", data_files={name: str(dataset_dir / f"{name}.jsonl") for name in ("train", "validation")})
    max_input, max_output = int(config["max_input_length"]), int(config["max_output_length"])

    def tokenize(batch):
        encoded = tokenizer(batch["instruction"], max_length=max_input, truncation=True)
        labels = tokenizer(text_target=batch["response"], max_length=max_output, truncation=True)
        encoded["labels"] = labels["input_ids"]
        return encoded

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)

    def compute_metrics(prediction):
        predictions, labels = prediction
        if isinstance(predictions, tuple): predictions = predictions[0]
        predictions = np.where(predictions < 0, tokenizer.pad_token_id, predictions)
        labels = np.where(labels < 0, tokenizer.pad_token_id, labels)
        decoded = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        valid = []
        for text in decoded:
            parsed, _, _ = parse_structured_output(text)
            valid.append(bool(parsed and all(theory.is_valid_roman(token) for token in parsed.roman_numerals)))
        return {"chord_validity": float(sum(valid) / max(1, len(valid)))}

    class LossValidityEarlyStopping(TrainerCallback):
        """Stop only after neither validation loss nor chord validity improves."""

        def __init__(self, patience: int):
            self.patience = patience
            self.best_loss = math.inf
            self.best_validity = -math.inf
            self.stale = 0

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            metrics = metrics or {}
            loss = float(metrics.get("eval_loss", math.inf))
            validity = float(metrics.get("eval_chord_validity", -math.inf))
            improved = loss < self.best_loss - 1e-6 or validity > self.best_validity + 1e-6
            if improved:
                self.best_loss = min(self.best_loss, loss)
                self.best_validity = max(self.best_validity, validity)
                self.stale = 0
            else:
                self.stale += 1
                if self.stale >= self.patience:
                    control.should_training_stop = True
            return control

    use_cuda = torch.cuda.is_available()
    precision = str(config.get("mixed_precision", "auto"))
    output_dir = Path(config["checkpoint_dir"]); output_dir.mkdir(parents=True, exist_ok=True)
    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir), num_train_epochs=float(config["epochs"]), max_steps=int(config["max_steps"]),
        per_device_train_batch_size=int(config["batch_size"]), per_device_eval_batch_size=int(config["batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]), learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]), warmup_ratio=float(config["warmup_ratio"]), lr_scheduler_type="linear",
        eval_strategy="steps", eval_steps=int(config["eval_steps"]), save_strategy="steps", save_steps=int(config["save_steps"]),
        logging_steps=int(config["logging_steps"]), save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False, predict_with_generate=True, generation_max_length=max_output,
        fp16=bool(use_cuda and precision in {"auto", "fp16"}), bf16=bool(use_cuda and precision == "bf16"),
        report_to=[] if not config.get("mlflow") else ["mlflow"], seed=seed, data_seed=seed,
    )
    if config.get("mlflow"):
        os.environ.setdefault("MLFLOW_TRACKING_URI", f"file:{Path('storage/mlruns').resolve().as_posix()}")
        os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "JDG GenAI Lyric-to-Chord Studio")
    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=tokenized["train"], eval_dataset=tokenized["validation"],
        processing_class=tokenizer, data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=compute_metrics, callbacks=[LossValidityEarlyStopping(int(config["early_stopping_patience"]))],
    )
    result = trainer.train(resume_from_checkpoint=resume_from)
    adapter_dir = Path(config["output_dir"]); adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    metrics = {key: float(value) if isinstance(value, (int, float)) else value for key, value in result.metrics.items()}
    report = {**parameter_report, "adapter_path": str(adapter_dir), "training_metrics": metrics, "seed": seed}
    write_json(adapter_dir / "training_report.json", report)
    if not (adapter_dir / "adapter_model.safetensors").exists():
        raise RuntimeError("PEFT adapter was not saved as Safetensors")
    return report


def merge_adapter(config_path: str | Path, destination: str | Path) -> str:
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    config = read_toml(config_path)
    base = AutoModelForSeq2SeqLM.from_pretrained(config["base_model"])
    adapted = PeftModel.from_pretrained(base, config["output_dir"])
    merged = adapted.merge_and_unload()
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(destination, safe_serialization=True)
    AutoTokenizer.from_pretrained(config["base_model"]).save_pretrained(destination)
    return str(destination)
