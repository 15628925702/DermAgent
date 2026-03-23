from memory.compressor import ExperienceCompressor
from memory.experience_bank import ExperienceBank
from memory.schema import build_raw_case_experience


def _raw_case(case_id: str, label: str, age: int, site: str) -> dict:
    return build_raw_case_experience(
        case_id=case_id,
        perception={
            "ddx_candidates": [{"name": label, "score": 0.8}],
            "uncertainty": {"level": "medium"},
            "visual_cues": ["pigmented"],
        },
        final_decision={"diagnosis": label, "final_label": label},
        selected_skills=["metadata_consistency_skill"],
        retrieval={},
        metadata={"age": age, "site": site},
    )


def test_compressor_adds_subgroup_prototypes_when_support_is_high():
    items = [
        _raw_case("c1", "NEV", 12, "trunk"),
        _raw_case("c2", "NEV", 13, "trunk"),
        _raw_case("c3", "NEV", 14, "trunk"),
        _raw_case("c4", "NEV", 15, "trunk"),
    ]
    bank = ExperienceBank(initial_items=items)

    summary = ExperienceCompressor().compress(bank, include_rules=False)
    prototypes = bank.get_prototypes()

    assert summary["prototype_count"] >= 2
    assert any(item.get("prototype_scope") == "global" for item in prototypes)
    assert any(item.get("prototype_scope") == "subgroup" for item in prototypes)
