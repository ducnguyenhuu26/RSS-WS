from scripts.aggregate_mujoco_results import get_nested


def test_get_nested_metric_from_result_payload():
    payload = {
        "program_residual": {
            "open_loop_bin_accuracy_h50": 0.25,
        }
    }

    assert get_nested(payload, "program_residual.open_loop_bin_accuracy_h50") == 0.25
