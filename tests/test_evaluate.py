"""Unit tests for evaluation metrics (SPEC §10). Hand-built synthetic forecasts."""

import math

from wc_model.evaluate import evaluate


def test_perfect_predictions_score_zero():
    predicted = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    actual = ["home", "draw", "away"]
    res = evaluate(predicted, actual)
    assert res["log_loss"] < 1e-9
    assert res["brier"] < 1e-9
    assert res["rps"] < 1e-9


def test_uniform_predictions_log_loss_is_ln3():
    predicted = [(1 / 3, 1 / 3, 1 / 3)] * 6
    actual = ["home", "draw", "away", "home", "draw", "away"]
    res = evaluate(predicted, actual)
    assert math.isclose(res["log_loss"], math.log(3), rel_tol=1e-9)


def test_better_than_market_scores_lower_log_loss():
    actual = ["home", "home", "draw", "away", "home"]
    # Model leans correctly toward the realized outcomes; market is flatter.
    model = [
        (0.7, 0.2, 0.1),
        (0.6, 0.25, 0.15),
        (0.25, 0.5, 0.25),
        (0.15, 0.25, 0.6),
        (0.65, 0.2, 0.15),
    ]
    market = [(1 / 3, 1 / 3, 1 / 3)] * 5
    res = evaluate(model, actual, market=market)
    assert res["log_loss"] < res["market"]["log_loss"]


def test_rps_rewards_predictions_closer_to_ordered_outcome():
    actual = ["away"]
    near = [(0.0, 0.2, 0.8)]   # mass near the realized (away) end
    far = [(0.8, 0.2, 0.0)]    # mass at the opposite (home) end
    rps_near = evaluate(near, actual)["rps"]
    rps_far = evaluate(far, actual)["rps"]
    assert rps_near < rps_far


def test_perfectly_calibrated_data_has_near_zero_calibration_error():
    # Constant forecast (0.5, 0.3, 0.2); outcomes occur at exactly those rates.
    predicted = [(0.5, 0.3, 0.2)] * 10
    actual = ["home"] * 5 + ["draw"] * 3 + ["away"] * 2
    res = evaluate(predicted, actual)
    assert res["calibration"]["calibration_error"] < 1e-9


def test_correct_score_log_loss_optional_branch():
    import numpy as np

    from wc_model.model import score_matrix

    p = score_matrix(0.2, 0.0, 0.0, 0.0, mu=0.0, gamma=0.0, rho=0.0, home=False)
    res = evaluate(
        [(0.5, 0.3, 0.2)],
        ["home"],
        scoreline_pred=[p],
        actual_scorelines=[(1, 0)],
    )
    assert math.isfinite(res["correct_score_log_loss"])
    assert math.isclose(res["correct_score_log_loss"], -math.log(float(p[1, 0])), rel_tol=1e-9)
