"""Fine-tune `numind/NuNER-multilingual-v0.1` on WikiANN-FR for PER/ORG/LOC."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Section 1: CLI arguments & config
# ---------------------------------------------------------------------------


DEFAULT_MODEL_NAME = "numind/NuNER-multilingual-v0.1"
DEFAULT_DATASET_NAME = "wikiann"
DEFAULT_DATASET_CONFIG = "fr"


@dataclass
class TrainingConfig:
    """Training configuration parsed from CLI arguments."""

    output_dir: Path = Path("models/ner_finetuned")
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    logging_steps: int = 200
    eval_steps: int = 400
    save_steps: int = 0  # 0 → use HF default (per epoch)
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None
    seed: int = 42
    model_name: str = DEFAULT_MODEL_NAME
    dataset_name: str = DEFAULT_DATASET_NAME
    dataset_config: str = DEFAULT_DATASET_CONFIG
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    notes: str = ""
    extra_tags: list[str] = field(default_factory=list)


def parse_args() -> TrainingConfig:
    """Parse CLI arguments into a TrainingConfig."""

    parser = argparse.ArgumentParser(
        description="Fine-tune NuNER on WikiANN-FR for PER/ORG/LOC tagging."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TrainingConfig.output_dir,
        help="Directory to store checkpoints and logs.",
    )
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainingConfig.weight_decay)
    parser.add_argument("--warmup-ratio", type=float, default=TrainingConfig.warmup_ratio)
    parser.add_argument("--logging-steps", type=int, default=TrainingConfig.logging_steps)
    parser.add_argument("--eval-steps", type=int, default=TrainingConfig.eval_steps)
    parser.add_argument("--save-steps", type=int, default=TrainingConfig.save_steps)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="If set, limit the number of training samples (useful for smoke tests).",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="If set, limit the number of evaluation samples.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training.",
    )
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument(
        "--model-name",
        type=str,
        default=TrainingConfig.model_name,
        help="Base model to fine-tune.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=TrainingConfig.dataset_name,
        help="HF dataset name (default: wikiann).",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=TrainingConfig.dataset_config,
        help="HF dataset config (default: fr).",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="If set, push the fine-tuned model to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="Model ID to use when pushing to the Hub.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional notes to include in metadata/logging.",
    )
    parser.add_argument(
        "--extra-tags",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of tags/labels for experiment tracking.",
    )

    args = parser.parse_args()
    return TrainingConfig(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        resume_from_checkpoint=args.resume_from_checkpoint,
        seed=args.seed,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        notes=args.notes,
        extra_tags=args.extra_tags or [],
    )


def main() -> None:
    """Entrypoint for the training script."""

    config = parse_args()
    print("Training configuration:")
    for field_name, value in config.__dict__.items():
        print(f"  {field_name}: {value}")

    # TODO: implement remaining sections
    raise SystemExit("Training pipeline not yet implemented.")


if __name__ == "__main__":
    main()



