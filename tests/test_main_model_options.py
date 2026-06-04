import torch

from main import (
    ZeroResidual,
    is_discrete_symbolic_model,
    model_uses_island_search,
    model_uses_neural_residual,
    model_uses_symbolic_gate,
    resolve_mujoco_dt,
    resolve_torch_device,
    symbolic_source_for_model,
)


def test_model_options_map_to_expected_components():
    assert symbolic_source_for_model("ours") == "llm"
    assert symbolic_source_for_model("ours_new") == "llm"
    assert symbolic_source_for_model("ours_gated") == "llm"
    assert symbolic_source_for_model("ours_gated_island") == "llm"
    assert symbolic_source_for_model("program_only") == "llm"
    assert symbolic_source_for_model("neural") == "empty"
    assert symbolic_source_for_model("symbolic") == "standard"
    assert symbolic_source_for_model("symbolic_neural") == "standard"

    assert model_uses_neural_residual("ours")
    assert model_uses_neural_residual("ours_new")
    assert model_uses_neural_residual("ours_gated")
    assert model_uses_neural_residual("ours_gated_island")
    assert model_uses_neural_residual("neural")
    assert model_uses_neural_residual("symbolic_neural")
    assert not model_uses_neural_residual("program_only")
    assert not model_uses_neural_residual("symbolic")
    assert model_uses_symbolic_gate("ours_new")
    assert model_uses_symbolic_gate("ours_gated")
    assert model_uses_symbolic_gate("ours_gated_island")
    assert not model_uses_symbolic_gate("ours")
    assert not model_uses_symbolic_gate("neural")
    assert model_uses_island_search("ours_new")
    assert model_uses_island_search("ours_gated_island")
    assert not model_uses_island_search("ours_gated")

    assert is_discrete_symbolic_model("discrete_symbolic")
    assert not is_discrete_symbolic_model("symbolic")
    assert not is_discrete_symbolic_model("ours")


def test_zero_residual_returns_zero_state_shaped_tensor():
    states = torch.ones(3, 4)
    zeros = ZeroResidual()(
        states,
        torch.ones(3, 2),
        torch.ones(3, 4),
        torch.ones(3, 4),
    )

    assert torch.equal(zeros, torch.zeros_like(states))


def test_resolve_mujoco_dt_accepts_numeric_override():
    assert resolve_mujoco_dt("Unused-v0", "0.0125") == 0.0125


def test_resolve_torch_device_auto_returns_available_device():
    device = resolve_torch_device("auto")

    assert device.type in {"cpu", "cuda"}
