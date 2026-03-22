import json

from agent.run_agent import run_agent
from integrations.openai_client import OpenAICompatClient


def test_smoke_run(monkeypatch):
    def fake_report(*args, **kwargs):
        return json.dumps(
            {
                "diagnosis": "NEV",
                "top_k": ["NEV", "SEK"],
                "reasoning": "offline smoke test",
                "evidence": ["fallback perception"],
                "risk_assessment": "low",
                "natural_language_report": "offline smoke test report",
            }
        )

    monkeypatch.setattr(OpenAICompatClient, "infer_derm_report", fake_report)

    case = {"file": "demo.png", "metadata": {}, "text": ""}
    result = run_agent(
        case,
        use_retrieval=False,
        use_reflection=False,
        use_controller=False,
        update_online=False,
    )
    assert "final_decision" in result
    assert result["report"]["diagnosis"] in {"NEV", "SEK", "ACK", "SCC", "BCC", "MEL", "UNKNOWN"}
