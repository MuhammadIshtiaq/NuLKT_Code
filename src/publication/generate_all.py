"""Generate all evidence-traceable publication artifacts."""

import json
from pathlib import Path

import pandas as pd

from src.evaluation.error_analysis import analyze_subgroups, write_error_analysis_report
from src.evaluation.claim_validator import ClaimValidator
from src.publication.bridge import export_publication_metrics
from src.publication.protocol import freeze_protocol
from src.publication.quality_control import generate_qc_report
from src.publication.reports import (
    generate_final_summary,
    generate_manuscript_notes,
    generate_novelty_analysis,
    generate_research_summary,
)
from src.publication.reproduction import build_reproduction_bundle
from src.publication.tables import generate_tables
from src.visualization.comparison_figures import generate_comparison_figures
from src.visualization.publication_figures import generate_available_figures


def generate(project_root: Path) -> None:
    results = project_root / "results"
    freeze_protocol(results)
    # Reshape experiment and analysis artifacts into the paths the table and
    # figure generators read. Sources that do not exist stay unavailable.
    export_publication_metrics(results)
    generate_available_figures(results)
    generate_comparison_figures(results)
    generate_tables(results)
    generate_final_summary(results)
    generate_manuscript_notes(results)
    generate_novelty_analysis(results)
    generate_research_summary(results)
    publication = results / "publication"
    prediction_path = results / "metrics" / "calibrated_test_predictions.csv"
    if prediction_path.exists():
        predictions = pd.read_csv(prediction_path)
        calibration = json.loads((results / "metrics" / "calibration_results.json").read_text(encoding="utf-8"))
        groups = [name for name in ("token_frequency_group", "linguistic_type", "history_group", "inactivity_group", "exercise_format", "cold_start", "uncertainty_group") if name in predictions]
        analysis = analyze_subgroups(predictions, groups, publication, calibration["threshold"])
    else:
        analysis = pd.DataFrame(columns=["subgroup_variable", "subgroup", "n", "status", "auroc", "auprc", "f1", "brier_score", "ece"])
        analysis.to_csv(publication / "subgroup_error_analysis.csv", index=False)
    write_error_analysis_report(analysis, publication / "error_analysis.md")
    validator = ClaimValidator(results)
    texts = "\n".join(path.read_text(encoding="utf-8") for path in publication.glob("*.md"))
    issues = validator.validate_evidence_contract() + validator.validate_text(texts)
    (publication / "claim_validation.json").write_text(
        json.dumps({"passed": not issues, "issues": [issue.__dict__ for issue in issues]}, indent=2),
        encoding="utf-8",
    )
    failed_runs = results / "failed_runs.json"
    if not failed_runs.exists():
        failed_runs.write_text(
            json.dumps(
                {
                    "status": "experiment_run_registry_unavailable",
                    "failed_runs": [],
                    "note": "No experiment-run registry was supplied; this is not evidence that all runs succeeded.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    build_reproduction_bundle(project_root)
    generate_qc_report(project_root)


if __name__ == "__main__":
    generate(Path.cwd())
