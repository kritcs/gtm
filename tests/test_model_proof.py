"""The Model & proof panel's data: badges present, never mixed, numbers consistent."""
import json
from pathlib import Path

import pytest

RES = Path(__file__).resolve().parents[1] / "sim" / "out" / "results.json"


@pytest.fixture(scope="module")
def d():
    if not RES.exists():
        pytest.skip("run `make sim` first")
    return json.loads(RES.read_text())


def test_every_section_has_one_known_badge(d):
    for key in ("market_model", "prior_residual", "thompson", "backtest", "ablation", "planted", "voice_dpo",
                "maturity", "measured"):
        assert d[key]["badge"] in ("synthetic", "simulated", "measured"), key


def test_voice_is_simulated_and_measured_is_empty(d):
    assert d["voice_dpo"]["badge"] == "simulated"
    assert d["measured"]["items"] == []


def test_ablation_matches_backtest(d):
    for k, v in d["ablation"]["methods"].items():
        assert v == d["backtest"]["methods"][k]


def test_planted_patterns_recovered(d):
    assert all(i["recovered"] for i in d["planted"]["items"])


def test_training_actually_ran(d):
    g = d["market_model"]["lightgcn"]["curve"]
    assert g[-1]["loss"] < g[0]["loss"]
    assert d["prior_residual"]["loss_curve"][-1]["loss"] < d["prior_residual"]["loss_curve"][0]["loss"]
