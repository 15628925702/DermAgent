from __future__ import annotations

import json
from typing import Any, Dict


def build_perception_prompt(metadata: Dict[str, Any]) -> str:
    metadata_text = json.dumps(metadata or {}, ensure_ascii=False, indent=2)

    return f"""
You are the perception module of a dermatology diagnosis agent.

Your task:
Given ONE skin lesion image and the provided metadata, return a structured JSON object.

Requirements:
1. Focus on visual dermatology perception, not long explanations.
2. Output ONLY valid JSON.
3. Diagnosis labels must be uppercase when possible, e.g. MEL, NEV, SCC, BCC, ACK, SEK.
4. Provide 3 to 5 differential diagnoses.
5. Include concise visual cues.
6. Include malignant_cues and suspicious_cues when applicable.
7. Estimate uncertainty.level as one of: low, medium, high.

Metadata:
{metadata_text}

Return JSON with exactly this schema:
{{
  "most_likely": {{
    "name": "SCC",
    "score": 0.62
  }},
  "ddx_candidates": [
    {{"name": "SCC", "score": 0.62}},
    {{"name": "ACK", "score": 0.25}},
    {{"name": "BCC", "score": 0.13}}
  ],
  "visual_cues": [
    "scaly erythematous plaque",
    "hyperkeratotic surface"
  ],
  "risk_cues": {{
    "malignant_cues": [
      "ulceration",
      "crusting"
    ],
    "suspicious_cues": [
      "sun-exposed site"
    ]
  }},
  "uncertainty": {{
    "level": "medium",
    "reason": "overlap between ACK and SCC"
  }}
}}

Rules:
- score must be numeric between 0 and 1
- ddx_candidates should be sorted descending by score
- use empty list if needed
- do not include markdown
""".strip()