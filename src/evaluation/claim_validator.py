"""Block research claims that are not traceable to machine-readable evidence."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ClaimIssue:
    severity: str
    rule: str
    excerpt: str
    message: str


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


class ClaimValidator:
    """Validate strong claims against statistical, metric, and literature files."""

    def __init__(self, results_root: Path, alpha: float = 0.05):
        self.root = results_root
        self.alpha = alpha
        self.statistics = _read_json(results_root / "statistical_tests" / "statistical_tests.json")
        self.main_results = self._load_csv(results_root / "metrics" / "main_results.csv")
        self.improvements = self._load_csv(results_root / "statistical_tests" / "computed_improvements.csv")
        self.published = self._load_csv(results_root / "published_studies.csv")
        self.sota_validation = _read_json(results_root / "statistical_tests" / "sota_validation.json")
        self.causal_evidence = _read_json(results_root / "causal" / "intervention_evidence.json")

    @staticmethod
    def _load_csv(path: Path) -> pd.DataFrame | None:
        try:
            return pd.read_csv(path) if path.exists() else None
        except Exception:
            return None

    @staticmethod
    def _normalized(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    def _significance_available(self, sentence: str) -> bool:
        records = self.statistics if isinstance(self.statistics, list) else (self.statistics or {}).get("comparisons", []) if isinstance(self.statistics, dict) else []
        for item in records:
            p = item.get("corrected_p_value")
            lo, hi = item.get("ci_lower"), item.get("ci_upper")
            if p is not None and p < self.alpha and lo is not None and hi is not None and not (lo <= 0 <= hi):
                model_a, model_b = self._normalized(str(item.get("model_a", ""))), self._normalized(str(item.get("model_b", "")))
                normalized_sentence = self._normalized(sentence)
                if model_a and model_b:
                    if model_a in normalized_sentence and model_b in normalized_sentence:
                        return True
                elif self.main_results is not None and "model" in self.main_results:
                    models = [self._normalized(str(value)) for value in self.main_results["model"].dropna().unique()]
                    if len(models) == 2 and all(model in normalized_sentence for model in models):
                        return True
        return False

    def _best_model_available(self, sentence: str = "") -> bool:
        if self.main_results is None or self.main_results.empty:
            return False
        required = {"model", "metric", "mean"}
        if not required.issubset(self.main_results.columns):
            return False
        frame = self.main_results.copy()
        frame["mean"] = pd.to_numeric(frame["mean"], errors="coerce")
        frame = frame.dropna(subset=["model", "metric", "mean"])
        if frame["model"].nunique() < 2:
            return False
        primary = frame[frame["metric"].astype(str).str.lower().isin(["auroc", "auprc"])]
        if primary.empty:
            return False
        winners = set()
        for _, group in primary.groupby(primary["metric"].astype(str).str.lower()):
            maximum = group["mean"].max()
            winners.update(group.loc[group["mean"].eq(maximum), "model"].astype(str))
        if len(winners) != 1:
            return False
        winner = next(iter(winners))
        return not sentence or self._normalized(winner) in self._normalized(sentence)

    def _improvement_available(self, value: float) -> bool:
        if self.improvements is None or "relative_improvement_percent" not in self.improvements:
            return False
        actual = pd.to_numeric(self.improvements["relative_improvement_percent"], errors="coerce").dropna().to_numpy()
        return bool(len(actual) and (abs(actual - value) <= 0.01).any())

    def _sota_available(self) -> bool:
        return bool(
            isinstance(self.sota_validation, dict)
            and self.sota_validation.get("validated") is True
            and self.sota_validation.get("directly_comparable") is True
            and self.sota_validation.get("protocol")
            and self.sota_validation.get("metric")
        )

    def validate_text(self, text: str) -> list[ClaimIssue]:
        issues: list[ClaimIssue] = []
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        for sentence in filter(str.strip, sentences):
            lowered = sentence.lower()
            cited_attribution = bool(
                re.search(r"\[[^\]]+\]", sentence)
                and re.search(
                    r"\b(?:authors?|organisers?|organizers?|study|paper|article)\b.{0,40}"
                    r"\b(?:report(?:s|ed)?|warn(?:s|ed)?|state(?:s|d)?)\b",
                    lowered,
                )
            )
            if (
                re.search(r"\b(significant(?:ly)?|statistically significant)\b", lowered)
                and not cited_attribution
                and not self._significance_available(sentence)
            ):
                issues.append(ClaimIssue("ERROR", "statistical_support", sentence, "Significance requires a corrected p-value and a confidence interval excluding the null."))
            if re.search(r"\b(best[- ]performing|performed best|highest performance)\b", lowered) and not self._best_model_available(sentence):
                issues.append(ClaimIssue("ERROR", "leaderboard_support", sentence, "Best-model language requires a complete machine-readable model comparison."))
            if re.search(r"\b(state[- ]of[- ]the[- ]art|sota)\b", lowered) and not self._sota_available():
                issues.append(ClaimIssue("ERROR", "sota_support", sentence, "SOTA language requires verified, directly comparable published studies and complete current results."))
            for match in re.finditer(
                r"\b(?:improv(?:e[ds]?|ement)|gain(?:ed)?)\b\D{0,25}"
                r"(\d+(?:\.\d+)?)\s*%",
                lowered,
            ):
                value = float(match.group(1))
                if not self._improvement_available(value):
                    issues.append(ClaimIssue("ERROR", "computed_improvement", sentence, f"The {value:g}% improvement is absent from computed_improvements.csv."))
            causal_instruction = re.search(r"explicit instruction.{0,50}(improv|caus|lead|result|increase|enhance)", lowered)
            limitation_language = re.search(r"\b(cannot|can't|does not|do not|not establish|no causal|without causal|withheld)\b", lowered)
            if causal_instruction and not limitation_language and not (isinstance(self.causal_evidence, dict) and self.causal_evidence.get("randomized_or_identified_causal_design") is True):
                issues.append(ClaimIssue("ERROR", "causal_instruction_claim", sentence, "SLAM is observational; causal instructional-effect claims require separate identified causal evidence."))
        return issues

    def validate_evidence_contract(self) -> list[ClaimIssue]:
        """Require the canonical comparison used by publication generators."""

        if self.main_results is None or self.main_results.empty:
            return [
                ClaimIssue(
                    "ERROR",
                    "main_results_contract",
                    "results/metrics/main_results.csv",
                    "Canonical machine-readable main results are missing or unreadable.",
                )
            ]
        required = {"model", "metric", "mean", "partition", "evaluation_unit", "source"}
        missing = sorted(required.difference(self.main_results.columns))
        if missing:
            return [
                ClaimIssue(
                    "ERROR",
                    "main_results_contract",
                    "results/metrics/main_results.csv",
                    f"Canonical main results are missing columns: {missing}.",
                )
            ]
        if self.main_results["model"].nunique() < 2:
            return [
                ClaimIssue(
                    "ERROR",
                    "main_results_contract",
                    "results/metrics/main_results.csv",
                    "Canonical main results must compare at least two measured models.",
                )
            ]
        values = pd.to_numeric(self.main_results["mean"], errors="coerce")
        if values.isna().any():
            return [
                ClaimIssue(
                    "ERROR",
                    "main_results_contract",
                    "results/metrics/main_results.csv",
                    "Canonical main results contain a missing or non-numeric metric value.",
                )
            ]
        return []

    def validate_file(self, manuscript: Path) -> list[ClaimIssue]:
        return self.validate_text(manuscript.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manuscript", type=Path)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/publication/claim_validation.json"))
    args = parser.parse_args()
    validator = ClaimValidator(args.results_root)
    issues = validator.validate_file(args.manuscript)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"passed": not issues, "issues": [asdict(issue) for issue in issues]}, indent=2), encoding="utf-8")
    if issues:
        raise SystemExit(f"Claim validation failed with {len(issues)} unsupported claim(s).")


if __name__ == "__main__":
    main()
