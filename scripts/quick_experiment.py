"""Run a real, deterministic smoke experiment on an official-data subset."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.torch_dataset import load_processed_frame, prepare_dataloaders
from src.evaluation.metrics import binary_classification_metrics
from src.models import run_classical_baseline
from src.training import run_neural_experiment
from src.utils.seeds import seed_everything
from src.visualization.evaluation_lines import generate_evaluation_line_graphs
from src.visualization.diagnostic_lines import generate_diagnostic_visualizations


WARNING = "DEBUG/QUICK MODE — NOT FOR PUBLICATION"
CLASSICAL_FEATURES = [
    "prior_interactions",
    "prior_error_rate",
    "recent_20_error_rate",
    "prior_token_exposures",
    "prior_token_error_rate",
    "log1p_elapsed_seconds",
    "part_of_speech_id",
    "morphology_id",
    "exercise_format_id",
]


def _predict(model: torch.nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            moved = {key: value.to(device) for key, value in batch.items()}
            output = model(moved, return_diagnostics=False)
            valid = ~moved["padding_mask"]
            probabilities.append(torch.sigmoid(output["logits"])[valid].cpu().numpy())
            targets.append(moved["responses"][valid].cpu().numpy())
    return np.concatenate(targets), np.concatenate(probabilities)


def _classical(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    train = frame[frame["chronological_split"].astype(str) == "train"]
    validation = frame[frame["chronological_split"].astype(str) == "validation"]
    if train.empty or validation.empty:
        raise RuntimeError("Quick subset lacks chronological train/validation rows")
    x_train = train[CLASSICAL_FEATURES].fillna(0).to_numpy(dtype=np.float32)
    x_validation = validation[CLASSICAL_FEATURES].fillna(0).to_numpy(dtype=np.float32)
    output = run_classical_baseline(
        "logistic_regression",
        x_train,
        train["label"].to_numpy(),
        x_validation,
        random_state=42,
        class_weight="balanced",
    )
    predictions = pd.DataFrame(
        {
            "label": validation["label"].to_numpy(dtype=np.int64),
            "probability": output["probabilities"],
        }
    )
    return (
        binary_classification_metrics(
            predictions["label"], predictions["probability"], threshold=0.5
        ),
        predictions,
    )


def run(root: Path) -> dict:
    seed_everything(42)
    output = root / "results" / "debug_quick"
    output.mkdir(parents=True, exist_ok=True)
    frame, manifest = load_processed_frame(root)
    baseline_metrics, baseline_predictions = _classical(frame)
    loaders, vocab_sizes = prepare_dataloaders(
        root,
        protocol="chronological_split",
        max_length=50,
        batch_size=16,
        num_workers=0,
    )
    if "validation" not in loaders:
        raise RuntimeError("Quick subset lacks a validation sequence loader")
    experiment = run_neural_experiment(
        "nulkt",
        vocab_sizes,
        loaders["train"],
        loaders["validation"],
        {
            "hidden_dim": 32,
            "embedding_dim": 16,
            "num_heads": 2,
            "local_layers": 1,
            "global_layers": 1,
            "num_experts": 3,
            "dropout": 0.1,
            "max_sequence_length": 50,
            "time_embedding_dim": 8,
            "learner_embedding_dim": 16,
        },
        {
            "epochs": 1,
            "patience": 1,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "optimizer": "adamw",
            "mixed_precision": False,
            "contrastive_weight": 0.01,
            "checkpoint_name": "quick_best",
        },
        output / "nulkt",
        device="cpu",
    )
    targets, probabilities = _predict(experiment["model"], loaders["validation"], torch.device("cpu"))
    proposed_predictions = pd.DataFrame({"label": targets, "probability": probabilities})
    proposed_metrics = binary_classification_metrics(targets, probabilities, threshold=0.5)
    result = {
        "warning": WARNING,
        "source": "official Harvard Dataverse English-learning track; deterministic preprocessing subset",
        "preprocessing_hash": manifest["preprocessing_hash"],
        "subset_interactions": manifest["interactions"],
        "subset_learners": manifest["learners"],
        "validation_only": True,
        "final_test_evaluated": False,
        "metrics": {
            "logistic_regression": baseline_metrics,
            "nulkt": proposed_metrics,
        },
    }
    (output / "quick_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    table = pd.DataFrame(
        [
            {"model": model, "auroc": values["auroc"], "auprc": values["auprc"], "f1": values["f1"]}
            for model, values in result["metrics"].items()
        ]
    )
    table.insert(0, "warning", WARNING)
    table.to_csv(output / "quick_table.csv", index=False)
    prediction_dir = output / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    baseline_predictions.to_csv(prediction_dir / "logistic_regression_validation.csv", index=False)
    proposed_predictions.to_csv(prediction_dir / "nulkt_validation.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    positions = np.arange(len(table))
    ax.bar(positions - 0.18, table["auroc"], width=0.36, label="AUROC", edgecolor="black")
    ax.bar(positions + 0.18, table["auprc"], width=0.36, label="AUPRC", edgecolor="black")
    ax.set_xticks(positions, table["model"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Validation metric")
    ax.legend(frameon=False)
    ax.set_title(WARNING, fontsize=10)
    fig.savefig(output / "quick_metrics.png", dpi=200)
    fig.savefig(output / "quick_metrics.pdf")
    plt.close(fig)
    generate_evaluation_line_graphs(
        {
            "Logistic Regression": baseline_predictions,
            "NuLKT": proposed_predictions,
        },
        output / "figures",
        title_suffix="Quick validation subset",
        watermark=WARNING,
        dpi=800,
    )
    generate_diagnostic_visualizations(
        {
            "Logistic Regression": baseline_predictions,
            "NuLKT": proposed_predictions,
        },
        output / "figures",
        title_suffix="Quick validation subset",
        watermark=WARNING,
        dpi=800,
    )
    return result


if __name__ == "__main__":
    raise SystemExit("Use scripts/reproduce_offline_safe.py; this legacy quick experiment is outside the corrected protocol.")
