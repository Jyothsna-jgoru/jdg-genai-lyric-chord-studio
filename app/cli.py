from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="jdg-studio", description="JDG GenAI Lyric-to-Chord Studio commands")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("dataset-generate", help="Generate a deterministic synthetic dataset")
    generate.add_argument("--config", default="configs/dataset_dev.toml")
    validate = sub.add_parser("dataset-validate", help="Validate dataset schemas, theory, and split leakage")
    validate.add_argument("--dataset-dir", default="data/generated/dev")
    train = sub.add_parser("train", help="Train and save a LoRA adapter")
    train.add_argument("--config", default="configs/train_dev.toml")
    resume = sub.add_parser("resume", help="Resume LoRA training from a Trainer checkpoint")
    resume.add_argument("--config", default="configs/train_dev.toml")
    resume.add_argument("--checkpoint", required=True)
    evaluate = sub.add_parser("evaluate", help="Evaluate base, LoRA, and theory baseline systems")
    evaluate.add_argument("--config", default="configs/train_dev.toml")
    evaluate.add_argument("--limit", type=int, default=None)
    infer = sub.add_parser("infer", help="Run one local neural inference request")
    infer.add_argument("--lyrics", required=True)
    infer.add_argument("--section", default="Verse")
    infer.add_argument("--key", default="C")
    infer.add_argument("--scale", default="major")
    infer.add_argument("--difficulty", default="beginner")
    infer.add_argument("--seed", type=int, default=42)
    infer.add_argument("--base-only", action="store_true")
    merge = sub.add_parser("merge", help="Optionally merge LoRA into a standalone model")
    merge.add_argument("--config", default="configs/train_dev.toml")
    merge.add_argument("--destination", required=True)
    serve = sub.add_parser("serve", help="Start the complete FastAPI application")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    if args.command == "dataset-generate":
        from app.ml.dataset import generate_dataset
        print(json.dumps(generate_dataset(args.config), indent=2))
    elif args.command == "dataset-validate":
        from app.ml.dataset import validate_dataset
        result = validate_dataset(args.dataset_dir); print(json.dumps(result, indent=2))
        if not result["valid"]: raise SystemExit(1)
    elif args.command in {"train", "resume"}:
        from app.ml.training import train_adapter
        print(json.dumps(train_adapter(args.config, args.checkpoint if args.command == "resume" else None), indent=2))
    elif args.command == "evaluate":
        from app.ml.evaluation import evaluate_systems
        print(json.dumps(evaluate_systems(args.config, args.limit), indent=2))
    elif args.command == "infer":
        from app.core import config as config_module
        from app.ml.dataset import build_instruction
        from app.ml.model import ModelService
        from app.schemas.studio import MusicalControls
        if args.base_only:
            object.__setattr__(config_module.settings, "adapter_path", Path("__missing_adapter__"))
        service = ModelService(); service.load(allow_download=True)
        controls = MusicalControls(key=args.key, scale=args.scale, difficulty=args.difficulty, seed=args.seed)
        instruction = build_instruction({"lyrics": args.lyrics, "section": args.section, "key": args.key, "scale": args.scale,
            "genre": "pop", "mood": "hopeful", "tempo": 96, "time_signature": "4/4", "difficulty": args.difficulty,
            "chord_density": 2, "variation": .35, "previous_progression": [], "lyric_features": {}})
        output, latency = service.generate(instruction, controls)
        print(json.dumps({"source": "lora_model_output" if service.adapter_loaded else "base_model_output", "raw_output": output,
                          "latency_ms": latency, "health": service.health()}, indent=2))
    elif args.command == "merge":
        from app.ml.training import merge_adapter
        print(merge_adapter(args.config, args.destination))
    else:
        import uvicorn
        uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()

