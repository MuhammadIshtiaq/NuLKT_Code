"""Run prespecified paired, learner-clustered comparisons from one prediction file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .statistics import holm_bonferroni, learner_clustered_bootstrap, learner_level_wilcoxon, mcnemar_exact


def compare_models(
    predictions: pd.DataFrame,
    reference: str,
    challengers: list[str],
    output_dir: Path,
    threshold: float,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> list[dict]:
    required = {"label", "learner_id", reference, *challengers}
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"prediction table is missing columns: {sorted(missing)}")
    records: list[dict] = []
    for challenger in challengers:
        for metric in ("auroc", "auprc"):
            item = learner_clustered_bootstrap(
                predictions["label"], predictions[challenger], predictions[reference], predictions["learner_id"],
                metric=metric, n_bootstrap=n_bootstrap, seed=seed,
            )
            records.append({"model_a": challenger, "model_b": reference, **item})
        wilcox = learner_level_wilcoxon(predictions["label"], predictions[challenger], predictions[reference], predictions["learner_id"])
        records.append({"model_a": challenger, "model_b": reference, "metric": "learner_brier_score", **wilcox})
        mcnemar = mcnemar_exact(predictions["label"], predictions[challenger] >= threshold, predictions[reference] >= threshold)
        records.append({"model_a": challenger, "model_b": reference, "metric": "accuracy_at_validation_threshold", "threshold": threshold, **mcnemar})
    correction = holm_bonferroni([row["raw_p_value"] for row in records])
    for row, adjusted, reject in zip(records, correction["adjusted_p_value"], correction["reject"]):
        row["corrected_p_value"] = adjusted
        row["reject_after_holm"] = reject
        row["correction_family"] = "all prespecified challenger-reference metric/test comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_dir / "statistical_tests.csv", index=False)
    (output_dir / "statistical_tests.json").write_text(json.dumps({"comparisons": records, "correction": "Holm-Bonferroni"}, indent=2), encoding="utf-8")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="CSV with label, learner_id, and one probability column per model")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--challenger", action="append", required=True)
    parser.add_argument("--threshold", type=float, required=True, help="Threshold selected on validation data")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results/statistical_tests"))
    args = parser.parse_args()
    compare_models(pd.read_csv(args.predictions), args.reference, args.challenger, args.output_dir, args.threshold, args.n_bootstrap, args.seed)


if __name__ == "__main__":
    main()
