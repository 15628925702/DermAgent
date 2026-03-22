from pathlib import Path

import datasets.splits as splits


def _build_cases(n: int):
    return [
        {"file": f"case_{idx}.png", "label": "BCC" if idx % 2 == 0 else "NEV"}
        for idx in range(n)
    ]


def test_load_or_create_split_manifest_rebuilds_stale_manifest(monkeypatch):
    old_cases = _build_cases(4)
    new_cases = _build_cases(8)
    stale_payload = splits.build_stratified_split(old_cases, seed=7)
    rebuilt_payload = splits.build_stratified_split(new_cases, seed=7)
    saved = {}

    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(splits, "load_split_manifest", lambda path: stale_payload)

    def fake_save(payload, path):
        saved["payload"] = payload
        saved["path"] = str(path)
        return Path(path)

    monkeypatch.setattr(splits, "save_split_manifest", fake_save)

    result = splits.load_or_create_split_manifest(new_cases, "dummy_split.json", seed=7)

    assert result["num_cases"] == 8
    assert saved["payload"]["num_cases"] == 8
    rebuilt_ids = {
        case_id
        for split_ids in result["splits"].values()
        for case_id in split_ids
    }
    assert rebuilt_ids == {case["file"] for case in new_cases}
    assert result == rebuilt_payload
