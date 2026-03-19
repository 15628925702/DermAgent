from agent.run_agent import run_agent


def test_smoke_run():
    case = {"file": "demo.png", "metadata": {}, "text": ""}
    result = run_agent(case)
    assert "final_decision" in result
