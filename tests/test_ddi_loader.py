import io
from pathlib import Path

import datasets.ddi as ddi
from datasets.ddi import (
    BENIGN_BINARY_LABEL,
    MALIGNANT_BINARY_LABEL,
    load_ddi_cases,
    map_ddi_binary_label,
    map_ddi_diagnosis_to_agent_label,
)


def test_ddi_mapping_helpers_cover_binary_and_agent_labels():
    assert map_ddi_diagnosis_to_agent_label("melanoma") == "MEL"
    assert map_ddi_diagnosis_to_agent_label("seborrheic keratosis") == "SEK"
    assert map_ddi_binary_label(diagnosis_value="basal cell carcinoma") == MALIGNANT_BINARY_LABEL
    assert map_ddi_binary_label(agent_label="NEV") == BENIGN_BINARY_LABEL


def test_load_ddi_cases_accepts_flexible_csv_columns(monkeypatch):
    csv_text = "\n".join(
        [
            "DDI_file,disease,malignant,fitzpatrick,skin_tone,body_site,patient_age,gender,partition",
            "case_a.jpg,melanoma,true,5,dark,arm,55,male,test",
            "case_b.png,nevus,false,2,light,back,18,female,test",
        ]
    )
    fake_csv = Path("fake_ddi_metadata.csv")

    monkeypatch.setattr(ddi, "_resolve_metadata_csv", lambda root, metadata_csv: fake_csv)
    monkeypatch.setattr(
        ddi,
        "_build_image_index",
        lambda root, images_dir: {
            "case_a.jpg": "images/case_a.jpg",
            "case_b.png": "images/case_b.png",
        },
    )
    monkeypatch.setattr(Path, "open", lambda self, *args, **kwargs: io.StringIO(csv_text))

    cases = load_ddi_cases("data/ddi", split_name="test")

    assert len(cases) == 2
    assert cases[0]["label"] == "MEL"
    assert cases[0]["binary_label"] == MALIGNANT_BINARY_LABEL
    assert cases[0]["metadata"]["fitzpatrick"] == "5"
    assert cases[1]["label"] == "NEV"
    assert cases[1]["binary_label"] == BENIGN_BINARY_LABEL
    assert Path(cases[1]["image_path"]).name == "case_b.png"
