"""Rule-based, auditable recommendations derived from model evidence.

This module recommends instructional *strategies*. It does not estimate or claim
the causal effectiveness of an intervention from observational SLAM data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class InstructionRecommendation:
    strategy: str
    priority: str
    rationale: tuple[str, ...]
    confidence: float
    teacher_review: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExplicitInstructionEngine:
    """Deterministic decision rules with configurable, serializable thresholds."""

    def __init__(
        self,
        error_threshold: float = 0.60,
        high_error_threshold: float = 0.80,
        mastery_threshold: float = 0.80,
        uncertainty_threshold: float = 0.45,
        forgetting_threshold: float = 0.60,
        minimum_exposures_for_retrieval: int = 3,
    ) -> None:
        for name, value in {
            "error_threshold": error_threshold,
            "high_error_threshold": high_error_threshold,
            "mastery_threshold": mastery_threshold,
            "uncertainty_threshold": uncertainty_threshold,
            "forgetting_threshold": forgetting_threshold,
        }.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        self.error_threshold = error_threshold
        self.high_error_threshold = high_error_threshold
        self.mastery_threshold = mastery_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.forgetting_threshold = forgetting_threshold
        self.minimum_exposures_for_retrieval = minimum_exposures_for_retrieval

    def recommend(self, evidence: Mapping[str, Any]) -> InstructionRecommendation:
        """Return one primary strategy from pre-computed learner evidence.

        Expected numeric fields are ``error_probability``, ``mastery``,
        ``uncertainty``, ``forgetting_signal``, and ``previous_exposures``.
        ``weakness`` may be lexical, grammar, morphology, or contextual.
        """

        p_error = self._probability(evidence, "error_probability")
        mastery = self._probability(evidence, "mastery")
        uncertainty = self._probability(evidence, "uncertainty")
        forgetting = self._probability(evidence, "forgetting_signal")
        exposures = max(0, int(evidence.get("previous_exposures", 0)))
        weakness = str(evidence.get("weakness", "contextual")).lower()
        historical_errors = max(0, int(evidence.get("historical_errors", 0)))

        if uncertainty >= self.uncertainty_threshold:
            return InstructionRecommendation(
                "Teacher/human review", "high",
                (f"predictive uncertainty {uncertainty:.3f} exceeds {self.uncertainty_threshold:.3f}",
                 "additional evidence is recommended before automated intervention"),
                1 - uncertainty, True,
            )
        if mastery >= self.mastery_threshold and p_error < self.error_threshold:
            return InstructionRecommendation(
                "Reduced scaffolding after mastery", "low",
                (f"mastery {mastery:.3f} meets threshold {self.mastery_threshold:.3f}",
                 f"predicted error risk is {p_error:.3f}"),
                1 - uncertainty,
            )
        if forgetting >= self.forgetting_threshold:
            strategy = "Retrieval practice" if exposures >= self.minimum_exposures_for_retrieval else "Spaced review"
            return InstructionRecommendation(
                strategy, "high" if p_error >= self.high_error_threshold else "medium",
                (f"forgetting signal {forgetting:.3f} exceeds {self.forgetting_threshold:.3f}",
                 f"prior exposures: {exposures}"), 1 - uncertainty,
            )
        if p_error >= self.error_threshold:
            strategy = {
                "lexical": "Explicit vocabulary explanation",
                "vocabulary": "Explicit vocabulary explanation",
                "grammar": "Explicit grammatical rule explanation",
                "grammatical": "Explicit grammatical rule explanation",
                "morphology": "Targeted morphological explanation",
                "morphological": "Targeted morphological explanation",
            }.get(weakness)
            if strategy is None:
                strategy = "Contrastive correct-versus-incorrect example" if historical_errors >= 2 else "Worked example"
            return InstructionRecommendation(
                strategy, "high" if p_error >= self.high_error_threshold else "medium",
                (f"predicted error risk {p_error:.3f} exceeds {self.error_threshold:.3f}",
                 f"dominant weakness: {weakness}", f"historical errors: {historical_errors}"),
                1 - uncertainty,
            )
        return InstructionRecommendation(
            "Guided practice", "low",
            (f"predicted error risk is {p_error:.3f}", "evidence does not trigger a more intensive strategy"),
            1 - uncertainty,
        )

    @staticmethod
    def _probability(evidence: Mapping[str, Any], field: str) -> float:
        value = float(evidence.get(field, 0.0))
        if not 0 <= value <= 1:
            raise ValueError(f"{field} must be in [0, 1], got {value}")
        return value

    def thresholds(self) -> dict[str, float | int]:
        return {
            "error_threshold": self.error_threshold,
            "high_error_threshold": self.high_error_threshold,
            "mastery_threshold": self.mastery_threshold,
            "uncertainty_threshold": self.uncertainty_threshold,
            "forgetting_threshold": self.forgetting_threshold,
            "minimum_exposures_for_retrieval": self.minimum_exposures_for_retrieval,
        }
