from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import DUCForwardOutput, DUCWorldModel


@dataclass(frozen=True)
class DUCLossConfig:
    beta_kl: float = 1e-3
    context_weight: float = 1.0
    residual_weight: float = 0.0
    control_weight: float = 0.0
    orth_weight: float = 0.0
    sparse_weight: float = 0.0
    unknown_weight: float = 0.0
    trust_region_weight: float = 1.0
    trust_region_delta_min: float = 0.15
    trust_region_delta_range: float = 0.75
    prior_beta_weight: float = 1e-4


@dataclass
class DUCLossOutput:
    total: torch.Tensor
    nll: torch.Tensor
    kl: torch.Tensor
    context: torch.Tensor
    residual: torch.Tensor
    control: torch.Tensor
    orth: torch.Tensor
    sparse: torch.Tensor
    unknown: torch.Tensor
    trust_region: torch.Tensor
    prior_beta: torch.Tensor


def kl_normal_diag(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_std: torch.Tensor,
) -> torch.Tensor:
    prior_var = prior_std.pow(2).clamp_min(1e-8)
    var = torch.exp(logvar)
    kl = 0.5 * (
        (var + (mean - prior_mean).pow(2)) / prior_var
        - 1.0
        + torch.log(prior_var)
        - logvar
    )
    return kl.sum(dim=-1).mean()


def weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None,
) -> torch.Tensor:
    error = (prediction - target).pow(2)
    if weights is not None:
        error = error * weights
    return error.mean()


def orthogonality_penalty(
    output: DUCForwardOutput,
    weights: torch.Tensor | None,
) -> torch.Tensor:
    effects = output.effects
    if weights is not None:
        effects = effects * weights.unsqueeze(1).sqrt()
    gram = torch.einsum("bkd,bld->bkl", effects, effects)
    eye = torch.eye(gram.shape[-1], device=gram.device, dtype=torch.bool)
    off_diag = gram[:, ~eye]
    return off_diag.pow(2).mean()


def residual_penalty(model: DUCWorldModel, output: DUCForwardOutput) -> torch.Tensor:
    confidences = model.prior_confidence.to(output.residual_effects.device)
    per_mechanism = output.residual_effects.pow(2).mean(dim=-1)
    # High-confidence law priors should need only small neural corrections.
    return (per_mechanism * (0.25 + confidences).unsqueeze(0)).mean()


def trust_region_penalty(
    model: DUCWorldModel,
    output: DUCForwardOutput,
    delta_min: float,
    delta_range: float,
) -> torch.Tensor:
    confidences = model.prior_confidence.to(output.residual_effects.device)
    prior_norm = output.prior_effects.pow(2).mean(dim=-1).add(1e-8).sqrt()
    residual_norm = output.residual_effects.pow(2).mean(dim=-1).add(1e-8).sqrt()
    delta = float(delta_min) + (1.0 - confidences) * float(delta_range)
    violation = torch.relu(residual_norm - delta.unsqueeze(0) * prior_norm)
    # Unknown mechanisms have confidence zero and should not be forced to stay
    # near a nonexistent law prior.
    weights = confidences.unsqueeze(0)
    return (weights * violation.pow(2)).mean()


def prior_beta_penalty(model: DUCWorldModel) -> torch.Tensor:
    # Keep law calibration near 1 unless data strongly needs a different scale.
    return model.prior_log_beta.pow(2).mean()


def unknown_activation_penalty(model: DUCWorldModel, output: DUCForwardOutput) -> torch.Tensor:
    if not model.unknown_indices:
        return output.alpha.new_zeros(())
    unknown_index = torch.tensor(model.unknown_indices, device=output.alpha.device)
    return output.alpha.index_select(dim=-1, index=unknown_index).abs().mean()


def compute_duc_loss(
    model: DUCWorldModel,
    output: DUCForwardOutput,
    targets: torch.Tensor,
    context_targets: torch.Tensor | None,
    config: DUCLossConfig,
    control_weights: torch.Tensor | None = None,
) -> DUCLossOutput:
    nll = model.nll(output, targets)
    kl = kl_normal_diag(
        output.posterior_mean,
        output.posterior_logvar,
        model.prior_mean.to(output.posterior_mean.device),
        model.prior_std.to(output.posterior_mean.device),
    )
    if context_targets is None:
        context = targets.new_zeros(())
    else:
        context = (output.alpha_mean - context_targets).pow(2).mean()
    residual = residual_penalty(model, output)
    control = weighted_mse(output.mean, targets, control_weights)
    orth = orthogonality_penalty(output, control_weights)
    sparse = output.alpha.abs().mean()
    unknown = unknown_activation_penalty(model, output)
    trust_region = trust_region_penalty(
        model,
        output,
        delta_min=config.trust_region_delta_min,
        delta_range=config.trust_region_delta_range,
    )
    prior_beta = prior_beta_penalty(model)
    total = (
        nll
        + config.beta_kl * kl
        + config.context_weight * context
        + config.residual_weight * residual
        + config.control_weight * control
        + config.orth_weight * orth
        + config.sparse_weight * sparse
        + config.unknown_weight * unknown
        + config.trust_region_weight * trust_region
        + config.prior_beta_weight * prior_beta
    )
    return DUCLossOutput(
        total=total,
        nll=nll.detach(),
        kl=kl.detach(),
        context=context.detach(),
        residual=residual.detach(),
        control=control.detach(),
        orth=orth.detach(),
        sparse=sparse.detach(),
        unknown=unknown.detach(),
        trust_region=trust_region.detach(),
        prior_beta=prior_beta.detach(),
    )
