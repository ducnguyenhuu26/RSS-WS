from scripts.aggregate_mujoco_results import get_nested


def test_get_nested_metric_from_result_payload():
    payload = {
        "score": {
            "duc_r2_at_1": 0.25,
        }
    }

    assert get_nested(payload, "score.duc_r2_at_1") == 0.25
