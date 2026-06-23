from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import DUCForwardOutput, DUCWorldModel


@dataclass(frozen=True)
class DUCLossConfig:
    beta_kl: float = 1e-3
    context_weight: float = 1.0
    control_weight: float = 0.0
    orth_weight: float = 0.0
    sparse_weight: float = 0.0


@dataclass
class DUCLossOutput:
    total: torch.Tensor
    nll: torch.Tensor
    kl: torch.Tensor
    context: torch.Tensor
    control: torch.Tensor
    orth: torch.Tensor
    sparse: torch.Tensor


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
    control = weighted_mse(output.mean, targets, control_weights)
    orth = orthogonality_penalty(output, control_weights)
    sparse = output.alpha.abs().mean()
    total = (
        nll
        + config.beta_kl * kl
        + config.context_weight * context
        + config.control_weight * control
        + config.orth_weight * orth
        + config.sparse_weight * sparse
    )
    return DUCLossOutput(
        total=total,
        nll=nll.detach(),
        kl=kl.detach(),
        context=context.detach(),
        control=control.detach(),
        orth=orth.detach(),
        sparse=sparse.detach(),
    )
