from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from app.core.config import settings


logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self) -> None:
        self.model: Any = None
        self.tokenizer: Any = None
        self.device = "cpu"
        self.base_loaded = False
        self.adapter_loaded = False
        self.adapter_version = "unavailable"
        self.load_error: str | None = None
        self.trainable_parameters = 0
        self.total_parameters = 0
        self._lock = threading.Lock()

    def load(self, allow_download: bool | None = None) -> None:
        if self.base_loaded:
            return
        allow_download = settings.allow_model_download if allow_download is None else allow_download
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(settings.base_model, local_files_only=not allow_download)
            base = AutoModelForSeq2SeqLM.from_pretrained(settings.base_model, local_files_only=not allow_download)
            base.to(self.device)
            adapter_config = settings.adapter_path / "adapter_config.json"
            if adapter_config.exists():
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(base, settings.adapter_path)
                self.adapter_loaded = True
                self.adapter_version = settings.adapter_path.name
            else:
                self.model = base
            self.model.eval()
            self.base_loaded = True
            self.total_parameters = sum(parameter.numel() for parameter in self.model.parameters())
            self.trainable_parameters = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
            self.load_error = None
            logger.info("Loaded local model base=%s adapter=%s device=%s", settings.base_model, self.adapter_loaded, self.device)
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Local model unavailable; theory fallback remains active: %s", self.load_error)

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self.base_loaded = False
        self.adapter_loaded = False

    def generate(self, instruction: str, controls: Any) -> tuple[str | None, float]:
        if not self.base_loaded:
            self.load()
        if not self.base_loaded:
            return None, 0.0
        import torch

        started = time.perf_counter()
        with self._lock, torch.inference_mode():
            torch.manual_seed(controls.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(controls.seed)
            encoded = self.tokenizer(instruction, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            options: dict[str, Any] = {"max_new_tokens": 256}
            if controls.decoding_method == "beam":
                options.update(num_beams=controls.num_beams, do_sample=False)
            elif controls.decoding_method == "temperature":
                options.update(do_sample=True, temperature=controls.temperature)
            elif controls.decoding_method == "top_k":
                options.update(do_sample=True, top_k=controls.top_k, temperature=controls.temperature)
            else:
                options.update(do_sample=False, num_beams=1)
            output = self.model.generate(**encoded, **options)
            raw = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return raw, round((time.perf_counter() - started) * 1000, 3)

    def health(self) -> dict[str, Any]:
        evaluation = None
        if settings.evaluation_path.exists():
            try:
                evaluation = json.loads(settings.evaluation_path.read_text(encoding="utf-8")).get("summary")
            except Exception:
                evaluation = None
        return {
            "base_model": settings.base_model, "base_model_status": "loaded" if self.base_loaded else "unavailable",
            "adapter_status": "loaded" if self.adapter_loaded else "unavailable", "adapter_version": self.adapter_version,
            "device": self.device, "trainable_parameter_count": self.trainable_parameters,
            "total_parameter_count": self.total_parameters, "last_evaluation_summary": evaluation,
            "fallback_engine_available": True, "load_error": self.load_error,
        }


model_service = ModelService()

