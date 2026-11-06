"""Official NuLKT fit with frozen offline histories and blind DEV inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import research_stages as stages  # noqa: E402
from src.analysis.inference import indexed_dataloader  # noqa: E402
from src.data.torch_dataset import load_processed_frame, vocab_sizes_for_model  # noqa: E402
from src.utils.seeds import seed_everything  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args()
    if (args.seed is None) == (not args.preflight):
        ap.error("Choose exactly one of --seed or --preflight")
    manifest_path = ROOT / "data/processed/en_es/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    check = ROOT / "offline_safe_preflight.json"
    if manifest.get("offline_safe_protocol") != "split-frozen-v1" or not check.is_file():
        raise RuntimeError("Safe data/preflight unavailable")
    if json.loads(check.read_text(encoding="utf-8"))["status"] != "PASS":
        raise RuntimeError("Safe data preflight failed")
    if args.preflight:
        dev, _ = load_processed_frame(
            ROOT, row_limit=512, derive_context=False,
            split_filter=("official_split", {"dev"}),
        )
        loader, _ = indexed_dataloader(
            ROOT, split="dev", protocol="official_split", frame=dev,
            sequence_length=200, batch_size=16, context_length=0,
        )
        batch = next(iter(loader))
        valid = ~batch["padding_mask"]
        if not torch.all(batch["observed_responses"][valid] == -1):
            raise RuntimeError("Held-out outcome visible in neural input")
        if not torch.all((batch["responses"][valid] == 0) | (batch["responses"][valid] == 1)):
            raise RuntimeError("True scoring targets unavailable")
        print(json.dumps({"status": "PASS", "safe_manifest": str(manifest_path),
                          "dev_rows_sampled": len(dev), "blind_response_positions": int(valid.sum())}), flush=True)
        return

    seed = args.seed
    tag = "offline_safe_nulkt"
    run_dir = ROOT / "results/raw_runs" / tag / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "status": "started", "seed": seed, "model": "nulkt",
        "partition": "official train for gradient updates; official dev for validation loss",
        "blind_dev_response_policy": "all valid DEV observed_responses=-1; labels retained only as targets",
        "test_policy": "not loaded during fit; post hoc after frozen combiner selection",
        "history_policy": manifest["offline_safe_semantics"],
        "prompt_registry": "official train only; unseen DEV/TEST prompt_code=0",
        "data_manifest_sha256": sha(manifest_path),
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (
            ROOT / "src/data/torch_dataset.py", ROOT / "src/models/nulkt.py",
            ROOT / "src/models/components.py", ROOT / "scripts/research_stages.py",
            ROOT / "src/training/trainer.py")},
        "epochs_ceiling": 100, "patience": 15,
        "model_overrides": {"exercise_context": True},
        "training_overrides": {"lr_schedule": "cosine_warmup", "warmup_ratio": 0.08,
                               "ema": True, "ema_decay": 0.9995},
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "safe_fit_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    # Push the official test predicate into Parquet reads.  No test target is
    # present in model-selection memory.
    frame, _ = load_processed_frame(
        ROOT, derive_context=False,
        split_filter=("official_split", {"train", "dev"}),
    )
    loaders = {}
    for split, alias in (("train", "train"), ("dev", "validation")):
        loader, _ = indexed_dataloader(
            ROOT, split=split, protocol="official_split", frame=frame,
            sequence_length=200, batch_size=64, context_length=0,
        )
        loaders[alias] = loader
    for split, alias in (("train", "train"), ("dev", "validation")):
        item = loaders[alias].dataset[0]
        valid = ~item["padding_mask"]
        if split == "dev" and not torch.all(item["observed_responses"][valid] == -1):
            raise RuntimeError("DEV response policy violated")
        if split == "train" and not torch.equal(item["observed_responses"][valid], item["responses"][valid]):
            raise RuntimeError("TRAIN response policy violated")
    seed_everything(seed)
    stages.PATIENCE_OVERRIDE = 15
    model, _, dev_metrics = stages._train(
        ROOT, loaders, vocab_sizes_for_model(ROOT), "nulkt", tag, seed, 100,
        model_overrides={"exercise_context": True},
        training_overrides={"lr_schedule": "cosine_warmup", "warmup_ratio": 0.08,
                            "ema": True, "ema_decay": 0.9995},
        sequence_length=200,
    )
    predictions = stages._predict(model, loaders["validation"], next(model.parameters()).device)
    path = run_dir / "dev_predictions.csv"
    predictions[["row_index", "label", "learner_id", "probability"]].to_csv(path, index=False)
    provenance.update({
        "status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "dev_metrics": dev_metrics, "dev_prediction_rows": len(predictions),
        "dev_prediction_sha256": sha(path),
    })
    (run_dir / "safe_fit_provenance.json").write_text(json.dumps(provenance, indent=2, default=float), encoding="utf-8")
    print(json.dumps({"status": "complete", "seed": seed, "dev_metrics": dev_metrics,
                      "prediction_rows": len(predictions)}, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
