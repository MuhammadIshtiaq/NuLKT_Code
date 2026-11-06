"""Generate an auditable PASS/WARNING/FAIL quality-control report."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Check:
    name: str
    status: str
    critical: bool
    evidence: str


def _exists(root: Path, relative: str, name: str, critical: bool = False) -> Check:
    path = root / relative
    return Check(name, "PASS" if path.exists() else ("FAIL" if critical else "WARNING"), critical, str(path) if path.exists() else f"Unavailable: {path}")


def _figure_check(root: Path) -> Check:
    registry = root / "figures" / "figure_registry.json"
    if not registry.exists():
        return Check("figures", "WARNING", False, "Figure registry unavailable")
    data = json.loads(registry.read_text(encoding="utf-8"))
    available = sum(item.get("status") == "available" for item in data.values())
    return Check("figures", "PASS" if available == len(data) else "WARNING", False, f"{available}/{len(data)} required figures available; absent-data figures were not fabricated")


def _registry_check(root: Path, relative: str, name: str, expected: int) -> Check:
    path = root / relative
    if not path.exists():
        return Check(name, "WARNING", False, f"Registry unavailable: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        numbered = [item for key, item in data.items() if str(key).isdigit()]
        available = sum(item.get("status") == "available" for item in numbered)
        return Check(name, "PASS" if available >= expected else "WARNING", False, f"{available}/{expected} required artifacts have evidence-backed content")
    except Exception as exc:
        return Check(name, "WARNING", False, f"Registry unreadable: {exc}")


def _json_boolean_check(root: Path, relative: str, name: str, field: str, critical: bool = False) -> Check:
    path = root / relative
    if not path.exists():
        return Check(name, "FAIL" if critical else "WARNING", critical, f"Unavailable: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        passed = data.get(field) is True
        return Check(name, "PASS" if passed else "FAIL", critical, f"{field}={data.get(field)!r} in {path}")
    except Exception as exc:
        return Check(name, "FAIL" if critical else "WARNING", critical, f"Unreadable JSON: {exc}")


def _csv_schema_check(
    root: Path,
    relative: str,
    name: str,
    required: set[str],
    *,
    critical: bool = False,
) -> Check:
    path = root / relative
    if not path.is_file():
        return Check(name, "FAIL" if critical else "WARNING", critical, f"Unavailable: {path}")
    try:
        import pandas as pd

        frame = pd.read_csv(path)
        missing = sorted(required.difference(frame.columns))
        valid = not frame.empty and not missing
        evidence = (
            f"{len(frame)} rows with required columns in {path}"
            if valid
            else f"Invalid or empty CSV; missing columns={missing}: {path}"
        )
        return Check(name, "PASS" if valid else "FAIL", critical, evidence)
    except Exception as exc:
        return Check(name, "FAIL" if critical else "WARNING", critical, f"Unreadable CSV: {exc}")


def _matched_ablation_check(root: Path) -> Check:
    path = root / "metrics" / "matched_ablation_results.csv"
    if not path.is_file():
        return Check(
            "five-seed matched-budget ablations",
            "WARNING",
            False,
            f"Unavailable: {path}",
        )
    try:
        import pandas as pd

        frame = pd.read_csv(path)
        required = {"variant", "seeds", "budget_epochs", "auroc_mean", "auroc_sd"}
        missing = sorted(required.difference(frame.columns))
        valid = (
            not frame.empty
            and not missing
            and (pd.to_numeric(frame["seeds"], errors="coerce") >= 5).all()
            and (pd.to_numeric(frame["budget_epochs"], errors="coerce") >= 30).all()
            and frame["variant"].astype(str).str.startswith("A0_full").any()
        )
        evidence = (
            f"{len(frame)} variants; >=5 seeds and >=30-epoch ceiling per variant: {path}"
            if valid
            else f"Incomplete matched protocol; missing columns={missing}: {path}"
        )
        return Check(
            "five-seed matched-budget ablations",
            "PASS" if valid else "WARNING",
            False,
            evidence,
        )
    except Exception as exc:
        return Check(
            "five-seed matched-budget ablations", "WARNING", False, f"Unreadable CSV: {exc}"
        )


def _prompt_control_check(root: Path) -> Check:
    path = root / "metrics" / "prompt_free_control.csv"
    protocol = root / "benchmark" / "prompt_free_control_protocol.json"
    if not path.is_file() or not protocol.is_file():
        return Check(
            "prompt-identity control",
            "WARNING",
            False,
            f"Unavailable result or frozen protocol: {path}; {protocol}",
        )
    try:
        import pandas as pd

        frame = pd.read_csv(path)
        pairs = set(zip(frame.get("model", []), frame.get("split", [])))
        expected = {
            ("full_features", "dev"),
            ("full_features", "test"),
            ("prompt_free", "dev"),
            ("prompt_free", "test"),
        }
        frozen = json.loads(protocol.read_text(encoding="utf-8"))
        valid = (
            expected.issubset(pairs)
            and "auroc" in frame
            and pd.to_numeric(frame["auroc"], errors="coerce").notna().all()
            and frozen.get("excluded_features") == ["prompt_id"]
            and frozen.get("changed_variables") == 1
        )
        return Check(
            "prompt-identity control",
            "PASS" if valid else "WARNING",
            False,
            f"Single-variable frozen control {'verified' if valid else 'incomplete'}: {path}",
        )
    except Exception as exc:
        return Check("prompt-identity control", "WARNING", False, f"Unreadable control: {exc}")


def _reproduction_check(root: Path) -> Check:
    path = root / "reproduction_bundle" / "manifest.json"
    if not path.exists():
        return Check("reproducibility", "WARNING", False, "Reproduction manifest unavailable")
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = sum(item.get("status") == "unavailable" for item in data.get("files", []))
    return Check("reproducibility", "PASS" if missing == 0 else "WARNING", False, f"Manifest available; {missing} required source categories unavailable")


def _font_check(root: Path) -> Check:
    path = root / "figures" / "font_report.json"
    if not path.exists():
        return Check("font requirement", "WARNING", False, "Font report unavailable")
    data = json.loads(path.read_text(encoding="utf-8"))
    return Check("font requirement", data.get("status", "WARNING"), False, f"Requested {data.get('requested')}; selected {data.get('selected')}")


def _failed_runs_check(root: Path) -> Check:
    path = root / "failed_runs.json"
    if not path.exists():
        return Check("failed-run disclosure", "WARNING", False, "Failed-run registry unavailable")
    data = json.loads(path.read_text(encoding="utf-8"))
    unavailable = data.get("status") == "experiment_run_registry_unavailable"
    return Check("failed-run disclosure", "WARNING" if unavailable else "PASS", False, data.get("note", str(path)))


def _integrity_check(root: Path, relative: str, name: str) -> Check:
    path = root / relative
    if not path.exists():
        return Check(name, "FAIL", True, f"Unavailable: {path}")
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            value = data.get("overall", data.get("status", data.get("passed")))
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            value = "PASS" if "Overall: PASS" in text else ("FAIL" if "Overall: FAIL" in text else None)
        passed = value is True or str(value).upper() in {"PASS", "PASSED", "AVAILABLE"}
        return Check(name, "PASS" if passed else "FAIL", True, f"Reported overall integrity state: {value!r}")
    except Exception as exc:
        return Check(name, "FAIL", True, f"Could not verify integrity artifact: {exc}")


def _dpi_check(root: Path) -> Check:
    """Confirm figures were written at the resolution this run declared.

    The check previously hard-coded 800 in both the filename it looked for and
    the threshold it applied, so a run rendered at any other resolution failed a
    requirement it was never trying to meet. It now compares against the
    resolution actually configured, which keeps the guarantee (figures are at
    the declared print resolution) without pinning it to one number.
    """
    from src.visualization.style import active_dpi

    required = active_dpi()
    name = f"{required}-DPI requirement"
    # Older runs wrote the resolution into the filename; accept both.
    candidates = [root / "figures" / "architecture.png", root / "figures" / "architecture_800dpi.png"]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return Check(name, "WARNING", False, "Architecture PNG unavailable")
    try:
        from PIL import Image

        with Image.open(path) as image:
            dpi = image.info.get("dpi", (0, 0))
        # One unit of tolerance: matplotlib writes 799.9984 for a requested 800.
        ok = min(dpi) >= required - 1
        return Check(name, "PASS" if ok else "FAIL", False, f"{path.name} metadata DPI={dpi}, required {required}")
    except Exception as exc:
        return Check(name, "WARNING", False, f"Could not inspect DPI: {exc}")


def run_checks(project_root: Path) -> list[Check]:
    results = project_root / "results"
    checks = [
        _exists(project_root, "data/metadata/dataset_metadata.json", "dataset download"),
        _integrity_check(results, "data_integrity_report.json", "dataset integrity"),
        _exists(results, "missing_data_report.json", "missing data"),
        _exists(results, "preprocessing_summary.json", "preprocessing"),
        _integrity_check(results, "data_leakage_audit.txt", "target and temporal leakage"),
        _integrity_check(results, "split_contamination_audit.json", "train/test contamination"),
        _exists(results, "test_evaluation_log.json", "test-set protection"),
        _reproduction_check(results),
        _exists(results, "metrics/seed_results.csv", "completed seeds"),
        _csv_schema_check(
            results,
            "metrics/main_results.csv",
            "canonical main results",
            {"model", "metric", "mean", "partition", "evaluation_unit", "source"},
            critical=True,
        ),
        _csv_schema_check(
            results,
            "metrics/canonical_ensemble_results.csv",
            "canonical ensemble study",
            {"combiner", "selected", "cv_auroc_mean", "test_auroc", "source"},
        ),
        _exists(results, "metrics/classical_validation_results.csv", "completed classical validation baselines"),
        _matched_ablation_check(results),
        _prompt_control_check(results),
        _exists(results, "metrics/optimizer_results.csv", "optimizer experiments"),
        _exists(results, "metrics/calibration_results.csv", "calibration"),
        _exists(results, "statistical_tests/statistical_tests.json", "statistical tests"),
        _figure_check(results),
        _dpi_check(results),
        _font_check(results),
        _registry_check(results, "tables/table_registry.json", "tables", 19),
        _exists(project_root, "references/related_work.bib", "references"),
        _failed_runs_check(results),
        _json_boolean_check(results, "publication/claim_validation.json", "unsupported research claims", "passed", critical=True),
    ]
    return checks


def generate_qc_report(project_root: Path) -> Path:
    checks = run_checks(project_root)
    critical_failure = any(item.critical and item.status == "FAIL" for item in checks)
    overall = "FAIL" if critical_failure else ("PASS" if all(item.status == "PASS" for item in checks) else "WARNING")
    colors = {"PASS": "#d9ead3", "WARNING": "#fff2cc", "FAIL": "#f4cccc"}
    rows = "".join(f"<tr><td>{html.escape(c.name)}</td><td style='background:{colors[c.status]}'>{c.status}</td><td>{'Yes' if c.critical else 'No'}</td><td>{html.escape(c.evidence)}</td></tr>" for c in checks)
    report = f"""<!doctype html><html><head><meta charset='utf-8'><title>Quality Control</title>
<style>body{{font-family:Arial,sans-serif;margin:2rem;color:#222}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #777;padding:.45rem;text-align:left}}th{{background:#e8eef5}}code{{font-family:monospace}}</style></head>
<body><h1>Research Pipeline Quality-Control Report</h1><p><strong>Overall status: {overall}</strong></p>
<p>WARNING means an experiment/artifact is currently unavailable; it is not silently treated as passed. A critical integrity failure prevents a successful-pipeline declaration.</p>
<table><thead><tr><th>Check</th><th>Status</th><th>Critical</th><th>Evidence</th></tr></thead><tbody>{rows}</tbody></table></body></html>"""
    destination = project_root / "results" / "publication" / "quality_control_report.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    (destination.parent / "quality_control_report.json").write_text(json.dumps({"overall": overall, "checks": [asdict(item) for item in checks]}, indent=2), encoding="utf-8")
    return destination


if __name__ == "__main__":
    generate_qc_report(Path.cwd())
