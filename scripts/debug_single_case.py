from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.run_agent import run_agent
from memory.experience_bank import ExperienceBank


def main() -> None:
    # 先改成你自己的测试图片路径
    case = {
        "file": "PAT_8_15_820",
        "image_path": "data/pad_ufes_20/imgs_part_1/PAT_8_15_820.png",
        "metadata": {
            "age": 53,
            "sex": "male",
            "site": "chest",
        },
        "text": "",
    }


    bank = ExperienceBank()
    result = run_agent(case=case, bank=bank)

    print("=" * 80)
    print("FINAL DECISION")
    print(json.dumps(result.get("final_decision", {}), ensure_ascii=False, indent=2))

    print("=" * 80)
    print("PERCEPTION")
    print(json.dumps(result.get("perception", {}), ensure_ascii=False, indent=2))

    print("=" * 80)
    print("REPORT")
    print(json.dumps(result.get("report", {}), ensure_ascii=False, indent=2))

    print("=" * 80)
    print("REFLECTION")
    print(json.dumps(result.get("reflection", {}), ensure_ascii=False, indent=2))

    print("=" * 80)
    print("TRACE")
    print(json.dumps(result.get("trace", []), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
