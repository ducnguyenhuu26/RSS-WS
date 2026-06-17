from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .core import ContinuousLawProtocol, ProgramOutput


class SymbolicProgram(nn.Module):
    """
    Executable continuous symbolic dynamics program.

    The program applies active laws to a single state-action pair, combines
    overlapping predictions as diagonal Gaussian product-of-experts, and marks
    dimensions with no confident symbolic prediction as unknown for the neural
    residual. Old laws that only return ``confidence`` remain valid: confidence
    is treated as a precision-like law weight.
    """

    def __init__(
        self,
        state_dim: int,
        laws: Sequence[ContinuousLawProtocol] | None = None,
        unknown_confidence_threshold: float = 1e-6,
        identity_for_unknown: bool = True,
        transition_dt: float = 1.0,
        composition_mode: str = "poe_next_state",
        learn_law_weights: bool = False,
        initial_law_logit: float = -8.0,
        base_delta_precision: float = 1.0,
    ) -> None:
        super().__init__()
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if transition_dt <= 0:
            raise ValueError("transition_dt must be positive")
        if composition_mode == "weighted_vector_field":
            composition_mode = "weighted_product_delta"
        if composition_mode not in {"poe_next_state", "weighted_product_delta"}:
            raise ValueError(
                "composition_mode must be 'poe_next_state' or "
                "'weighted_product_delta'"
            )
        if base_delta_precision < 0:
            raise ValueError("base_delta_precision must be nonnegative")
        self.state_dim = int(state_dim)
        self.unknown_confidence_threshold = float(unknown_confidence_threshold)
        self.identity_for_unknown = bool(identity_for_unknown)
        self.transition_dt = float(transition_dt)
        self.composition_mode = str(composition_mode)
        self.learn_law_weights = bool(learn_law_weights)
        self.base_delta_precision = float(base_delta_precision)

        module_laws: list[nn.Module] = []
        for law in laws or []:
            if not isinstance(law, nn.Module):
                raise TypeError(
                    "SymbolicProgram laws must inherit torch.nn.Module "
                    "so trainable symbolic parameters are registered."
                )
            module_laws.append(law)
        self.laws = nn.ModuleList(module_laws)
        if self.learn_law_weights:
            self.law_logits = nn.Parameter(
                torch.full((len(module_laws),), float(initial_law_logit))
            )
        else:
            self.register_buffer("law_logits", torch.zeros(len(module_laws)))

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> ProgramOutput:
        return self.predict(states, actions)

    def predict(self, states: torch.Tensor, actions: torch.Tensor) -> ProgramOutput:
        squeeze = states.ndim == 1
        states_batched = states.unsqueeze(0) if squeeze else states
        actions_batched = actions.unsqueeze(0) if actions.ndim == 1 else actions

        if states_batched.ndim != 2:
            raise ValueError("states must have shape [state_dim] or [batch, state_dim]")
        if actions_batched.ndim != 2:
            raise ValueError(
                "actions must have shape [action_dim] or [batch, action_dim]"
            )
        if states_batched.shape[0] != actions_batched.shape[0]:
            raise ValueError("states and actions must have the same batch size")
        if states_batched.shape[1] != self.state_dim:
            raise ValueError(
                f"expected state_dim={self.state_dim}, got {states_batched.shape[1]}"
            )

        outputs = [
            self._predict_one(state, action)
            for state, action in zip(states_batched, actions_batched)
        ]
        next_state = torch.stack([output.next_state for output in outputs], dim=0)
        confidence = torch.stack([output.confidence for output in outputs], dim=0)
        unknown_mask = torch.stack([output.unknown_mask for output in outputs], dim=0)
        variance = torch.stack(
            [
                output.variance
                if output.variance is not None
                else torch.full_like(output.confidence, float("inf"))
                for output in outputs
            ],
            dim=0,
        )
        active_laws = tuple(output.active_laws[0] for output in outputs)

        if squeeze:
            next_state = next_state.squeeze(0)
            confidence = confidence.squeeze(0)
            unknown_mask = unknown_mask.squeeze(0)
            variance = variance.squeeze(0)

        return ProgramOutput(
            next_state=next_state,
            confidence=confidence,
            unknown_mask=unknown_mask,
            active_laws=active_laws,
            variance=variance,
        )

    def _predict_one(self, state: torch.Tensor, action: torch.Tensor) -> ProgramOutput:
        if self.composition_mode == "weighted_product_delta":
            return self._predict_one_weighted_product_delta(state, action)
        return self._predict_one_poe_next_state(state, action)

    def _predict_one_poe_next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> ProgramOutput:
        numerator = torch.zeros(self.state_dim, dtype=state.dtype, device=state.device)
        denominator = torch.zeros_like(numerator)
        active_laws: list[str] = []

        for law in self.laws:
            if not law.precondition(state, action):  # type: ignore[attr-defined]
                continue
            prediction = law.predict(state, action)  # type: ignore[attr-defined]
            indices = prediction.indices.to(device=state.device, dtype=torch.long)
            values = prediction.values.to(device=state.device, dtype=state.dtype)
            confidence = prediction.confidence.to(
                device=state.device,
                dtype=state.dtype,
            )
            precision = _prediction_precision(
                prediction=prediction,
                confidence=confidence,
                device=state.device,
                dtype=state.dtype,
            )
            if indices.ndim != 1:
                raise ValueError("law prediction indices must be one-dimensional")
            if (
                values.shape != indices.shape
                or confidence.shape != indices.shape
                or precision.shape != indices.shape
            ):
                raise ValueError(
                    "law prediction values/confidence/precision must match indices"
                )
            if torch.any(indices < 0) or torch.any(indices >= self.state_dim):
                raise ValueError("law prediction index out of bounds")

            numerator.index_add_(0, indices, values * precision)
            denominator.index_add_(0, indices, precision)
            active_laws.append(prediction.law_name)

        known = denominator > self.unknown_confidence_threshold
        if self.identity_for_unknown:
            next_state = state.clone()
        else:
            next_state = torch.zeros_like(state)
        next_state = torch.where(
            known,
            numerator / denominator.clamp_min(1e-12),
            next_state,
        )
        unknown_mask = (~known).to(dtype=state.dtype)
        variance = torch.where(
            known,
            denominator.clamp_min(1e-12).reciprocal(),
            torch.full_like(denominator, float("inf")),
        )

        return ProgramOutput(
            next_state=next_state,
            confidence=denominator,
            unknown_mask=unknown_mask,
            active_laws=(tuple(active_laws),),
            variance=variance,
        )

    def _predict_one_weighted_product_delta(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> ProgramOutput:
        numerator = torch.zeros(self.state_dim, dtype=state.dtype, device=state.device)
        law_precision = torch.zeros_like(numerator)
        active_laws: list[str] = []

        for law_idx, law in enumerate(self.laws):
            if not law.precondition(state, action):  # type: ignore[attr-defined]
                continue
            prediction = law.predict(state, action)  # type: ignore[attr-defined]
            indices = prediction.indices.to(device=state.device, dtype=torch.long)
            values = prediction.values.to(device=state.device, dtype=state.dtype)
            confidence = prediction.confidence.to(
                device=state.device,
                dtype=state.dtype,
            )
            precision = _prediction_precision(
                prediction=prediction,
                confidence=confidence,
                device=state.device,
                dtype=state.dtype,
            )
            if indices.ndim != 1:
                raise ValueError("law prediction indices must be one-dimensional")
            if (
                values.shape != indices.shape
                or confidence.shape != indices.shape
                or precision.shape != indices.shape
            ):
                raise ValueError(
                    "law prediction values/confidence/precision must match indices"
                )
            if torch.any(indices < 0) or torch.any(indices >= self.state_dim):
                raise ValueError("law prediction index out of bounds")

            law_weight = self._law_weight(law_idx, dtype=state.dtype, device=state.device)
            effective_weight = precision * law_weight
            local_delta = _prediction_to_delta(
                prediction=prediction,
                state=state,
                indices=indices,
                values=values,
                transition_dt=self.transition_dt,
            )
            numerator.index_add_(0, indices, local_delta * effective_weight)
            law_precision.index_add_(0, indices, effective_weight)
            active_laws.append(prediction.law_name)

        base_precision = torch.full_like(law_precision, self.base_delta_precision)
        denominator = base_precision + law_precision
        combined_delta = numerator / denominator.clamp_min(1e-12)
        known = law_precision > self.unknown_confidence_threshold
        next_state = state + combined_delta

        unknown_mask = (~known).to(dtype=state.dtype)
        variance = torch.where(
            denominator > 0,
            denominator.clamp_min(1e-12).reciprocal(),
            torch.full_like(denominator, float("inf")),
        )

        return ProgramOutput(
            next_state=next_state,
            confidence=law_precision,
            unknown_mask=unknown_mask,
            active_laws=(tuple(active_laws),),
            variance=variance,
        )

    def _law_weight(
        self,
        law_idx: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if not self.learn_law_weights:
            return torch.ones((), dtype=dtype, device=device)
        return torch.nn.functional.softplus(
            self.law_logits[law_idx].to(dtype=dtype, device=device)
        )

    def symbolic_weight_l1(self) -> torch.Tensor:
        if len(self.laws) == 0 or not self.learn_law_weights:
            return self.law_logits.sum() * 0.0
        return torch.nn.functional.softplus(self.law_logits).sum()


def _prediction_to_delta(
    prediction,
    state: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
    transition_dt: float,
) -> torch.Tensor:
    value_kind = str(getattr(prediction, "value_kind", "delta")).lower()
    if value_kind == "next_state":
        return values - state[indices]
    if value_kind == "delta":
        return values
    if value_kind == "rate":
        return values * float(transition_dt)
    raise ValueError(
        f"Unsupported LawPrediction.value_kind={value_kind!r}; "
        "expected 'next_state', 'delta', or 'rate'"
    )


def _prediction_precision(
    prediction,
    confidence: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    precision = confidence.clamp_min(0.0)
    weight = getattr(prediction, "weight", None)
    if weight is not None:
        precision = torch.as_tensor(weight, device=device, dtype=dtype).clamp_min(0.0)
    std = getattr(prediction, "std", None)
    if std is not None:
        std_tensor = torch.as_tensor(std, device=device, dtype=dtype).abs().clamp_min(1e-6)
        precision = precision / std_tensor.square()
    return precision
