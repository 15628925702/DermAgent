from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


def sigmoid(value: float) -> float:
    value = max(-20.0, min(20.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


@dataclass
class SkillSpec:
    skill_id: str
    description: str
    when_to_use: str
    constraints: List[str] = field(default_factory=list)
    category: str = "generic"
    threshold: float = 0.5
    priority: float = 0.0
    routable: bool = True
    embedding: List[float] = field(default_factory=list)
    feature_weights: Dict[str, float] = field(default_factory=dict)
    stats: Dict[str, float] = field(
        default_factory=lambda: {
            "uses": 0.0,
            "helpful": 0.0,
            "updates": 0.0,
        }
    )

    def logit(self, features: Dict[str, float], extra_bias: float = 0.0) -> float:
        score = float(self.priority) + float(extra_bias)
        for name, weight in self.feature_weights.items():
            score += float(weight) * float(features.get(name, 0.0))
        return score

    def probability(self, features: Dict[str, float], extra_bias: float = 0.0) -> float:
        return sigmoid(self.logit(features, extra_bias=extra_bias))

    def update_weights(
        self,
        features: Dict[str, float],
        target: float,
        prediction: float,
        learning_rate: float,
    ) -> None:
        error = float(target) - float(prediction)
        self.priority += learning_rate * error
        for name, value in features.items():
            current = float(self.feature_weights.get(name, 0.0))
            self.feature_weights[name] = current + learning_rate * error * float(value)
        self.stats["updates"] = float(self.stats.get("updates", 0.0)) + 1.0

    def record_use(self, helpful: bool) -> None:
        self.stats["uses"] = float(self.stats.get("uses", 0.0)) + 1.0
        if helpful:
            self.stats["helpful"] = float(self.stats.get("helpful", 0.0)) + 1.0

    def success_rate(self) -> float:
        uses = float(self.stats.get("uses", 0.0))
        if uses <= 0:
            return 0.5
        return float(self.stats.get("helpful", 0.0)) / uses

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "description": self.description,
            "when_to_use": self.when_to_use,
            "constraints": list(self.constraints),
            "category": self.category,
            "threshold": self.threshold,
            "priority": round(self.priority, 4),
            "routable": self.routable,
            "embedding": list(self.embedding),
            "feature_weights": {
                key: round(float(value), 4)
                for key, value in sorted(self.feature_weights.items())
            },
            "stats": {
                key: round(float(value), 4)
                for key, value in sorted(self.stats.items())
            },
            "success_rate": round(self.success_rate(), 4),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillSpec":
        stats = data.get("stats", {}) or {}
        return cls(
            skill_id=str(data.get("skill_id", "")).strip(),
            description=str(data.get("description", "")).strip(),
            when_to_use=str(data.get("when_to_use", "")).strip(),
            constraints=[
                str(x).strip()
                for x in data.get("constraints", []) or []
                if str(x).strip()
            ],
            category=str(data.get("category", "generic")).strip() or "generic",
            threshold=float(data.get("threshold", 0.5)),
            priority=float(data.get("priority", 0.0)),
            routable=bool(data.get("routable", True)),
            embedding=[float(x) for x in data.get("embedding", []) or []],
            feature_weights={
                str(key): float(value)
                for key, value in (data.get("feature_weights", {}) or {}).items()
            },
            stats={
                "uses": float(stats.get("uses", 0.0)),
                "helpful": float(stats.get("helpful", 0.0)),
                "updates": float(stats.get("updates", 0.0)),
            },
        )


class SkillIndex:
    def __init__(self, specs: Optional[Iterable[SkillSpec]] = None) -> None:
        self._specs: Dict[str, SkillSpec] = {}
        for spec in specs or []:
            self.register(spec)

    def register(self, spec: SkillSpec) -> None:
        self._specs[spec.skill_id] = spec

    def get(self, skill_id: str) -> Optional[SkillSpec]:
        return self._specs.get(skill_id)

    def all_specs(self) -> List[SkillSpec]:
        return list(self._specs.values())

    def routable_specs(self) -> List[SkillSpec]:
        return [spec for spec in self._specs.values() if spec.routable]

    def as_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            skill_id: spec.to_dict()
            for skill_id, spec in sorted(self._specs.items())
        }

    def load_dict(self, payload: Dict[str, Dict[str, Any]]) -> None:
        self._specs = {}
        for _, spec_payload in sorted((payload or {}).items()):
            spec = SkillSpec.from_dict(spec_payload)
            if spec.skill_id:
                self.register(spec)

    @classmethod
    def from_dict(cls, payload: Dict[str, Dict[str, Any]]) -> "SkillIndex":
        index = cls()
        index.load_dict(payload)
        return index


def build_default_skill_index() -> SkillIndex:
    specs = [
        SkillSpec(
            skill_id="uncertainty_assessment_skill",
            description="Estimate uncertainty and keep a controller-side calibration anchor.",
            when_to_use="Nearly always useful as a lightweight calibration skill.",
            constraints=["Should not override diagnosis directly."],
            category="calibration",
            threshold=0.45,
            priority=0.9,
            embedding=[0.9, 0.2, 0.1],
            feature_weights={
                "bias": 0.9,
                "uncertainty_high": 0.7,
                "uncertainty_medium": 0.4,
                "retrieval_low": 0.2,
            },
        ),
        SkillSpec(
            skill_id="compare_skill",
            description="Directly compare the top ambiguous candidates.",
            when_to_use="When top candidates are close, uncertainty is high, or confusion memory exists.",
            constraints=["Best when at least two plausible candidates are available."],
            category="comparison",
            threshold=0.55,
            priority=-0.15,
            embedding=[0.2, 0.9, 0.2],
            feature_weights={
                "bias": -0.15,
                "uncertainty_high": 1.2,
                "uncertainty_medium": 0.6,
                "top_gap_small": 1.1,
                "has_confusion_support": 0.7,
                "retrieval_low": 0.35,
                "top_candidate_count": 0.2,
            },
        ),
        SkillSpec(
            skill_id="malignancy_risk_skill",
            description="Estimate malignant tendency and produce a safety-oriented label preference.",
            when_to_use="When malignant labels or malignant cues are present.",
            constraints=["Should primarily act as a safety constraint."],
            category="risk",
            threshold=0.5,
            priority=-0.2,
            embedding=[0.8, 0.8, 0.1],
            feature_weights={
                "bias": -0.2,
                "has_malignant_candidate": 1.15,
                "malignant_candidate_ratio": 0.8,
                "num_malignant_cues": 0.9,
                "num_suspicious_cues": 0.4,
                "strong_invasive_history": 0.55,
            },
        ),
        SkillSpec(
            skill_id="metadata_consistency_skill",
            description="Use age, site, and history priors to validate or penalize labels.",
            when_to_use="When metadata is available and visual or retrieval evidence is unstable.",
            constraints=["Only useful when metadata exists."],
            category="consistency",
            threshold=0.52,
            priority=-0.3,
            embedding=[0.3, 0.3, 0.9],
            feature_weights={
                "bias": -0.3,
                "metadata_present": 1.0,
                "retrieval_low": 0.75,
                "supports_top1": -0.45,
                "uncertainty_high": 0.45,
                "uncertainty_medium": 0.25,
                "strong_invasive_history": 0.15,
            },
        ),
        SkillSpec(
            skill_id="ack_scc_specialist_skill",
            description="Resolve ACK vs SCC confusion with specialist priors.",
            when_to_use="When ACK/SCC pair is present or confusion memory points to this pair.",
            constraints=["Useful only when SCC is plausible or metadata resembles sun-exposed keratinizing lesions."],
            category="specialist",
            threshold=0.56,
            priority=-0.45,
            embedding=[0.7, 0.6, 0.4],
            feature_weights={
                "bias": -0.45,
                "has_ack_scc_pair": 1.55,
                "has_confusion_support": 0.55,
                "sun_exposed_site": 0.45,
                "retrieval_recommends_ack_scc": 0.8,
                "num_malignant_cues": 0.15,
            },
        ),
        SkillSpec(
            skill_id="bcc_scc_specialist_skill",
            description="Resolve BCC vs SCC confusion with morphology and site-aware criteria.",
            when_to_use="When BCC and SCC are both plausible or confusion memory highlights this pair.",
            constraints=["Useful only when both keratinizing malignancy candidates are plausible."],
            category="specialist",
            threshold=0.56,
            priority=-0.42,
            embedding=[0.8, 0.55, 0.35],
            feature_weights={
                "bias": -0.42,
                "has_bcc_scc_pair": 1.6,
                "has_confusion_support": 0.45,
                "retrieval_recommends_bcc_scc": 0.85,
                "has_malignant_candidate": 0.25,
                "num_malignant_cues": 0.22,
                "sun_exposed_site": 0.15,
                "strong_invasive_history": 0.2,
                "top_gap_small": 0.3,
            },
        ),
        SkillSpec(
            skill_id="bcc_sek_specialist_skill",
            description="Resolve BCC vs SEK confusion using benign-versus-basal-cell features.",
            when_to_use="When BCC and SEK are both plausible or confusion memory highlights this pair.",
            constraints=["Useful only when a benign keratosis alternative remains plausible."],
            category="specialist",
            threshold=0.56,
            priority=-0.42,
            embedding=[0.55, 0.45, 0.75],
            feature_weights={
                "bias": -0.42,
                "has_bcc_sek_pair": 1.7,
                "has_confusion_support": 0.4,
                "retrieval_recommends_bcc_sek": 0.85,
                "metadata_present": 0.15,
                "sun_exposed_site": 0.18,
                "top_gap_small": 0.28,
                "uncertainty_high": 0.2,
            },
        ),
        SkillSpec(
            skill_id="mel_nev_specialist_skill",
            description="Resolve melanoma vs nevus confusion with targeted criteria.",
            when_to_use="When MEL and NEV are both plausible or confusion memory highlights this pair.",
            constraints=["Useful only when melanocytic ambiguity exists."],
            category="specialist",
            threshold=0.56,
            priority=-0.45,
            embedding=[0.7, 0.4, 0.7],
            feature_weights={
                "bias": -0.45,
                "has_mel_nev_pair": 1.65,
                "has_confusion_support": 0.45,
                "retrieval_recommends_mel_nev": 0.8,
                "uncertainty_high": 0.35,
                "top_gap_small": 0.45,
            },
        ),
    ]
    return SkillIndex(specs)
