from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from memory.experience_bank import ExperienceBank
from memory.schema import (
    build_confusion_experience,
    build_prototype_experience,
    build_rule_experience,
)


class ExperienceCompressor:
    def __init__(
        self,
        *,
        min_cases_per_prototype: int = 1,
        min_cases_per_confusion: int = 1,
        min_cases_per_rule: int = 3,
    ) -> None:
        self.min_cases_per_prototype = max(1, int(min_cases_per_prototype))
        self.min_cases_per_confusion = max(1, int(min_cases_per_confusion))
        self.min_cases_per_rule = max(1, int(min_cases_per_rule))

    def compress(self, bank: ExperienceBank, *, include_rules: bool = True) -> Dict[str, Any]:
        raw_cases = bank.get_raw_cases()
        hard_cases = bank.get_hard_cases()
        prototypes = self._build_prototypes(raw_cases)
        confusions = self._build_confusions(raw_cases)
        rules = self._build_rules(hard_cases) if include_rules else []

        prototype_update = bank.replace_type("prototype", prototypes)
        confusion_update = bank.replace_type("confusion", confusions)
        if include_rules:
            rule_update = bank.replace_type("rule", rules)
        else:
            rule_update = {"removed": 0, "added": 0, "skipped": 1}

        return {
            "raw_case_count": len(raw_cases),
            "hard_case_count": len(hard_cases),
            "prototype_count": len(prototypes),
            "confusion_count": len(confusions),
            "rule_count": len(rules),
            "prototype_labels": [item.get("disease", "") for item in prototypes],
            "confusion_pairs": [item.get("pair", []) for item in confusions],
            "rule_names": [item.get("rule_name", "") for item in rules],
            "prototype_update": prototype_update,
            "confusion_update": confusion_update,
            "rule_update": rule_update,
            "rules_enabled": include_rules,
        }

    def _build_prototypes(self, raw_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in raw_cases:
            label = self._extract_final_label(item)
            if label:
                groups[label].append(item)

        prototypes: List[Dict[str, Any]] = []
        for label, cases in sorted(groups.items()):
            if len(cases) < self.min_cases_per_prototype:
                continue

            cues = self._top_strings(
                cue
                for case in cases
                for cue in case.get("perception", {}).get("visual_cues", []) or []
            )
            common_confusions = self._top_strings(
                candidate
                for case in cases
                for candidate in self._other_ddx_names(case, exclude=label)
            )
            recommended_skills = self._top_strings(
                skill
                for case in cases
                for skill in case.get("selected_skills", []) or []
            )

            prototype = build_prototype_experience(
                disease=label,
                typical_cues=cues[:8],
                typical_metadata=self._summarize_metadata(cases),
                common_confusions=common_confusions[:5],
                recommended_skills=recommended_skills[:5],
            )
            prototype["source_count"] = len(cases)
            prototype["source_case_ids"] = [
                str(case.get("case_id", "")).strip() for case in cases[:10] if str(case.get("case_id", "")).strip()
            ]
            prototype["compression_level"] = self._support_level(len(cases))
            prototypes.append(prototype)

        return prototypes

    def _build_confusions(self, raw_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        ordered_pairs: Dict[str, List[str]] = {}

        for item in raw_cases:
            if not self._is_confusion_candidate(item):
                continue
            pair = self._extract_top2_pair(item)
            if len(pair) != 2:
                continue
            key = "||".join(pair)
            groups[key].append(item)
            ordered_pairs[key] = pair

        confusions: List[Dict[str, Any]] = []
        for key, cases in sorted(groups.items()):
            if len(cases) < self.min_cases_per_confusion:
                continue

            pair = ordered_pairs[key]
            clues = self._top_strings(
                cue
                for case in cases
                for cue in case.get("perception", {}).get("visual_cues", []) or []
            )
            useful_skills = self._filter_skills_for_pair(
                pair,
                self._top_strings(
                    skill
                    for case in cases
                    for skill in case.get("selected_skills", []) or []
                ),
            )
            failure_modes = self._top_strings(
                mode
                for case in cases
                for mode in self._derive_failure_modes(case)
            )

            confusion = build_confusion_experience(
                disease_a=pair[0],
                disease_b=pair[1],
                distinguishing_clues=clues[:8],
                useful_skills=useful_skills[:6],
                failure_modes=failure_modes[:6],
            )
            confusion["source_count"] = len(cases)
            confusion["source_case_ids"] = [
                str(case.get("case_id", "")).strip() for case in cases[:10] if str(case.get("case_id", "")).strip()
            ]
            confusion["label_votes"] = dict(
                Counter(self._extract_final_label(case) for case in cases if self._extract_final_label(case))
            )
            confusion["compression_level"] = self._support_level(len(cases))
            confusions.append(confusion)

        return confusions

    def _build_rules(self, hard_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rules: List[Dict[str, Any]] = []
        rules.extend(self._build_pair_rules(hard_cases))
        rules.extend(self._build_weak_support_rules(hard_cases))
        rules.extend(self._build_label_rules(hard_cases))
        return rules

    def _build_pair_rules(self, hard_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for item in hard_cases:
            pair = self._extract_top2_pair_from_hard_case(item)
            if len(pair) == 2:
                groups[(pair[0], pair[1])].append(item)

        rules: List[Dict[str, Any]] = []
        for pair, cases in sorted(groups.items()):
            if len(cases) < self.min_cases_per_rule:
                continue

            suggested_skills = self._filter_skills_for_pair(
                pair,
                self._top_strings(
                    skill
                    for case in cases
                    for skill in case.get("selected_skills", []) or []
                    if skill != "uncertainty_assessment_skill"
                ),
            )
            if not suggested_skills:
                continue

            rule = build_rule_experience(
                rule_name=f"rule_pair_{pair[0].lower()}_{pair[1].lower()}",
                trigger_conditions={
                    "min_uncertainty_level": "medium",
                    "requires_all_diseases": list(pair),
                },
                action={
                    "suggested_skills": suggested_skills[:4],
                    "target_pair": list(pair),
                    "reason": f"Repeated hard cases with pair {pair[0]}/{pair[1]}",
                },
                priority=min(4, 1 + len(cases)),
            )
            rule["source_count"] = len(cases)
            rule["source_case_ids"] = [
                str(case.get("case_id", "")).strip() for case in cases[:10] if str(case.get("case_id", "")).strip()
            ]
            rule["compression_level"] = self._support_level(len(cases))
            rules.append(rule)
        return rules

    def _build_weak_support_rules(self, hard_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        weak_cases = [
            case for case in hard_cases
            if bool((case.get("learning_signals", {}) or {}).get("low_retrieval_support", False))
        ]
        if len(weak_cases) < self.min_cases_per_rule + 1:
            return []

        suggested_skills = self._top_strings(
            skill
            for case in weak_cases
            for skill in case.get("selected_skills", []) or []
            if skill in {"compare_skill", "metadata_consistency_skill", "malignancy_risk_skill"}
        )
        if len(suggested_skills) < 2:
            return []

        top_labels = self._top_strings(
            label
            for case in weak_cases
            for label in case.get("top_ddx", []) or []
            if str(label).strip().upper() != "UNKNOWN"
        )
        if not top_labels:
            return []

        rule = build_rule_experience(
            rule_name="rule_weak_support_backoff",
            trigger_conditions={
                "min_uncertainty_level": "high",
                "requires_any_disease": top_labels[:3],
            },
            action={
                "suggested_skills": suggested_skills[:3],
                "reason": "Repeated low-support hard cases benefit from compare plus metadata/risk checks",
            },
            priority=2,
        )
        rule["source_count"] = len(weak_cases)
        rule["source_case_ids"] = [
            str(case.get("case_id", "")).strip() for case in weak_cases[:10] if str(case.get("case_id", "")).strip()
        ]
        rule["compression_level"] = self._support_level(len(weak_cases))
        return [rule]

    def _build_label_rules(self, hard_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in hard_cases:
            label = str(item.get("final_label", "")).strip().upper()
            if label and label != "UNKNOWN":
                groups[label].append(item)

        rules: List[Dict[str, Any]] = []
        for label, cases in sorted(groups.items()):
            if len(cases) < self.min_cases_per_rule + 1:
                continue

            selected_skills = self._filter_skills_for_label(
                label,
                self._top_strings(
                    skill
                    for case in cases
                    for skill in case.get("selected_skills", []) or []
                    if skill != "uncertainty_assessment_skill"
                ),
            )
            if len(selected_skills) < 2:
                continue

            dominant_pair = self._top_strings(
                "|".join(self._extract_top2_pair_from_hard_case(case))
                for case in cases
                if len(self._extract_top2_pair_from_hard_case(case)) == 2
            )
            rule = build_rule_experience(
                rule_name=f"rule_label_{label.lower()}_hardcase",
                trigger_conditions={
                    "min_uncertainty_level": "medium",
                    "requires_any_disease": [label],
                },
                action={
                    "suggested_skills": selected_skills[:3],
                    "reason": f"Hard cases ending as {label} repeatedly used the same recovery skills",
                    "dominant_pairs": dominant_pair[:2],
                },
                priority=2,
            )
            rule["source_count"] = len(cases)
            rule["source_case_ids"] = [
                str(case.get("case_id", "")).strip() for case in cases[:10] if str(case.get("case_id", "")).strip()
            ]
            rule["compression_level"] = self._support_level(len(cases))
            rules.append(rule)
        return rules

    def _filter_skills_for_pair(self, pair: Tuple[str, str] | List[str], skills: List[str]) -> List[str]:
        pair_set = {str(x).strip().upper() for x in pair}
        allowed = {"compare_skill", "metadata_consistency_skill", "malignancy_risk_skill"}
        if pair_set == {"ACK", "SCC"}:
            allowed.add("ack_scc_specialist_skill")
        if pair_set == {"MEL", "NEV"}:
            allowed.add("mel_nev_specialist_skill")
        if pair_set == {"BCC", "SEK"}:
            allowed.add("bcc_sek_specialist_skill")
        if pair_set == {"BCC", "SCC"}:
            allowed.add("bcc_scc_specialist_skill")
        return [skill for skill in skills if skill in allowed]

    def _filter_skills_for_label(self, label: str, skills: List[str]) -> List[str]:
        label = str(label).strip().upper()
        allowed = {"compare_skill", "metadata_consistency_skill", "malignancy_risk_skill"}
        if label in {"ACK", "SCC"}:
            allowed.add("ack_scc_specialist_skill")
        if label in {"MEL", "NEV"}:
            allowed.add("mel_nev_specialist_skill")
        if label in {"BCC", "SEK"}:
            allowed.add("bcc_sek_specialist_skill")
        if label in {"BCC", "SCC"}:
            allowed.add("bcc_scc_specialist_skill")
        return [skill for skill in skills if skill in allowed]

    def _summarize_metadata(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        ages = [age for age in (self._safe_int((case.get("metadata", {}) or {}).get("age")) for case in cases) if age is not None]
        sex = self._most_common(self._norm((case.get("metadata", {}) or {}).get("sex")) for case in cases)
        site = self._most_common(
            self._norm(
                (case.get("metadata", {}) or {}).get("location")
                or (case.get("metadata", {}) or {}).get("site")
                or (case.get("metadata", {}) or {}).get("anatomical_site")
            )
            for case in cases
        )

        metadata: Dict[str, Any] = {
            "sex": sex,
            "location": site,
            "site": site,
            "age_group": self._summarize_age_group(ages),
        }
        if ages:
            metadata["age"] = round(sum(ages) / len(ages), 1)
            metadata["age_min"] = min(ages)
            metadata["age_max"] = max(ages)
        return metadata

    def _derive_failure_modes(self, item: Dict[str, Any]) -> List[str]:
        modes: List[str] = []
        tags = item.get("tags", {}) or {}
        retrieval_summary = item.get("retrieval_summary", {}) or {}
        selected_skills = set(item.get("selected_skills", []) or [])

        if str(tags.get("uncertainty_level", "")).lower() == "high":
            modes.append("high_uncertainty")
        if tags.get("fallback_reason"):
            modes.append("fallback_case")
        if int(retrieval_summary.get("num_raw_case_hits", 0)) == 0 and int(retrieval_summary.get("num_prototype_hits", 0)) == 0:
            modes.append("weak_retrieval_support")
        if int(retrieval_summary.get("num_confusion_hits", 0)) == 0:
            modes.append("no_confusion_memory")
        if "compare_skill" in selected_skills:
            modes.append("needed_compare_skill")
        if any(skill in selected_skills for skill in {"mel_nev_specialist_skill", "ack_scc_specialist_skill"}):
            modes.append("needed_specialist_skill")

        return modes

    def _is_confusion_candidate(self, item: Dict[str, Any]) -> bool:
        pair = self._extract_top2_pair(item)
        if len(pair) != 2:
            return False

        tags = item.get("tags", {}) or {}
        retrieval_summary = item.get("retrieval_summary", {}) or {}
        selected_skills = set(item.get("selected_skills", []) or [])

        return (
            str(tags.get("uncertainty_level", "")).lower() in {"medium", "high"}
            or "compare_skill" in selected_skills
            or "mel_nev_specialist_skill" in selected_skills
            or "ack_scc_specialist_skill" in selected_skills
            or int(retrieval_summary.get("num_confusion_hits", 0)) > 0
        )

    def _extract_top2_pair(self, item: Dict[str, Any]) -> List[str]:
        ddx = item.get("perception", {}).get("ddx", []) or []
        names: List[str] = []
        for candidate in ddx:
            name = str(candidate.get("name", "")).strip().upper()
            if name and name not in names:
                names.append(name)
            if len(names) >= 2:
                break
        if len(names) < 2:
            return []
        return sorted(names[:2])

    def _extract_top2_pair_from_hard_case(self, item: Dict[str, Any]) -> List[str]:
        top_ddx = [str(x).strip().upper() for x in item.get("top_ddx", []) if str(x).strip()]
        names: List[str] = []
        for name in top_ddx[:2]:
            if name and name not in names:
                names.append(name)
        return sorted(names) if len(names) == 2 else []

    def _extract_final_label(self, item: Dict[str, Any]) -> str:
        final_decision = item.get("final_decision", {}) or {}
        for key in ["final_label", "diagnosis"]:
            label = str(final_decision.get(key, "")).strip().upper()
            if label:
                return label
        return ""

    def _other_ddx_names(self, item: Dict[str, Any], *, exclude: str) -> List[str]:
        exclude = str(exclude).strip().upper()
        names: List[str] = []
        for candidate in item.get("perception", {}).get("ddx", []) or []:
            name = str(candidate.get("name", "")).strip().upper()
            if not name or name == exclude or name in names:
                continue
            names.append(name)
        return names

    def _top_strings(self, values: Iterable[Any], limit: int = 8) -> List[str]:
        counter: Counter[str] = Counter()
        for value in values:
            text = str(value).strip()
            if text:
                counter[text] += 1
        return [name for name, _ in counter.most_common(limit)]

    def _most_common(self, values: Iterable[str]) -> str:
        items = [value for value in values if value]
        if not items:
            return ""
        return Counter(items).most_common(1)[0][0]

    def _summarize_age_group(self, ages: List[int]) -> str:
        if not ages:
            return ""
        avg_age = sum(ages) / len(ages)
        if avg_age < 30:
            return "young"
        if avg_age < 60:
            return "middle"
        return "older"

    def _support_level(self, count: int) -> str:
        if count >= 5:
            return "high"
        if count >= 2:
            return "medium"
        return "low"

    def _safe_int(self, value: Any) -> int | None:
        try:
            if value is None or value == "":
                return None
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _norm(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip().lower()

