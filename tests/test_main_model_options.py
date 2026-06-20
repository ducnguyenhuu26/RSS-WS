import torch

from main import (
    ZeroResidual,
    is_discrete_symbolic_model,
    is_dreamer_v3_model,
    is_pets_ensemble_model,
    model_uses_island_search,
    model_uses_neural_residual,
    model_uses_ode_residual,
    model_uses_symbolic_gate,
    normalize_model_name,
    residual_backbone_name,
    resolve_mujoco_dt,
    resolve_torch_device,
    symbolic_source_for_model,
)


def test_model_options_map_to_expected_components():
    assert normalize_model_name("ANSWER") == "answer"
    assert normalize_model_name("answer-mlp") == "answer_mlp"
    assert normalize_model_name("neural_ode") == "neural"
    assert normalize_model_name("DreamerV3") == "dreamer_v3"
    assert symbolic_source_for_model("answer") == "llm"
    assert symbolic_source_for_model("answer_mlp") == "llm"
    assert symbolic_source_for_model("program_only") == "llm"
    assert symbolic_source_for_model("neural") == "empty"
    assert symbolic_source_for_model("neural_mlp") == "empty"
    assert symbolic_source_for_model("symbolic_neural") == "standard"

    assert model_uses_neural_residual("answer")
    assert model_uses_neural_residual("answer_mlp")
    assert model_uses_neural_residual("neural")
    assert model_uses_neural_residual("neural_mlp")
    assert model_uses_neural_residual("symbolic_neural")
    assert not model_uses_neural_residual("program_only")
    assert model_uses_symbolic_gate("answer")
    assert model_uses_symbolic_gate("answer_mlp")
    assert not model_uses_symbolic_gate("neural")
    assert model_uses_island_search("answer")
    assert model_uses_island_search("answer_mlp")
    assert not model_uses_island_search("program_only")
    assert model_uses_ode_residual("answer")
    assert model_uses_ode_residual("neural")
    assert model_uses_ode_residual("symbolic_neural")
    assert not model_uses_ode_residual("answer_mlp")
    assert not model_uses_ode_residual("neural_mlp")
    assert residual_backbone_name("answer") == "ode"
    assert residual_backbone_name("answer_mlp") == "mlp"
    assert residual_backbone_name("neural") == "ode"
    assert residual_backbone_name("neural_mlp") == "mlp"

    assert is_discrete_symbolic_model("discrete_symbolic")
    assert not is_discrete_symbolic_model("answer")
    assert is_pets_ensemble_model("pets_ensemble")
    assert not is_pets_ensemble_model("neural")
    assert is_dreamer_v3_model("dreamer_v3")
    assert is_dreamer_v3_model("DreamerV3")
    assert not is_dreamer_v3_model("pets_ensemble")


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
