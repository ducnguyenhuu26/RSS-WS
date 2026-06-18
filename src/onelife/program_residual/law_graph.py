from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .core import ContinuousLawProtocol, TransitionBatch
from .task_specs import get_mujoco_task_spec


@dataclass(frozen=True)
class ConceptNode:
    """Semantic leader node used to ground executable follower laws."""

    name: str
    kind: str
    dimension_indices: tuple[int, ...] = ()
    action_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class LawNode:
    """Executable follower law node in the concept-law-dimension DAG."""

    name: str
    law_index: int
    parent_concepts: tuple[str, ...]
    target_dims: tuple[int, ...]
    quality: float
    origin: str = "symbolic_law"


@dataclass(frozen=True)
class LawGraph:
    """Typed concept -> law -> state-dimension DAG for symbolic interventions."""

    concepts: tuple[ConceptNode, ...]
    laws: tuple[LawNode, ...]
    concept_to_law_edges: tuple[tuple[str, str], ...]
    law_to_dim_edges: tuple[tuple[str, int], ...]
    dimension_budget: tuple[float, ...]

    def to_summary(self) -> dict[str, object]:
        return {
            "num_concepts": len(self.concepts),
            "num_laws": len(self.laws),
            "num_concept_to_law_edges": len(self.concept_to_law_edges),
            "num_law_to_dim_edges": len(self.law_to_dim_edges),
            "dimension_budget": list(self.dimension_budget),
            "laws": [
                {
                    "name": law.name,
                    "law_index": law.law_index,
                    "parent_concepts": list(law.parent_concepts),
                    "target_dims": list(law.target_dims),
                    "quality": law.quality,
                    "origin": law.origin,
                }
                for law in self.laws
            ],
        }


def build_law_graph(
    laws: Sequence[ContinuousLawProtocol],
    batch: TransitionBatch,
    env_id: str | None,
    transition_dt: float = 1.0,
    validation_sample_count: int = 256,
    quality_floor: float = 0.05,
) -> LawGraph:
    state_dim = int(batch.states.shape[1])
    action_dim = int(batch.actions.shape[1])
    concepts = _concept_nodes(env_id, state_dim, action_dim)
    concepts_by_name = {concept.name: concept for concept in concepts}
    dimension_concepts = _dimension_concept_names(concepts)
    action_concepts = _action_concept_names(concepts)

    law_nodes: list[LawNode] = []
    concept_to_law_edges: list[tuple[str, str]] = []
    law_to_dim_edges: list[tuple[str, int]] = []
    budget = [0.0 for _ in range(state_dim)]

    for law_index, law in enumerate(laws):
        law_name = _law_name(law, law_index)
        target_dims = _law_target_dims(law, batch, validation_sample_count)
        if not target_dims:
            continue
        quality = _law_quality(
            law,
            batch,
            target_dims,
            transition_dt,
            validation_sample_count,
            quality_floor=quality_floor,
        )
        parent_concepts = _law_parent_concepts(
            law_name=law_name,
            target_dims=target_dims,
            dimension_concepts=dimension_concepts,
            action_concepts=action_concepts,
            concepts_by_name=concepts_by_name,
        )
        node = LawNode(
            name=law_name,
            law_index=law_index,
            parent_concepts=parent_concepts,
            target_dims=target_dims,
            quality=quality,
        )
        law_nodes.append(node)
        for concept in parent_concepts:
            concept_to_law_edges.append((concept, law_name))
        for dim in target_dims:
            law_to_dim_edges.append((law_name, dim))
            budget[dim] = 1.0 - (1.0 - budget[dim]) * (1.0 - quality)

    return LawGraph(
        concepts=tuple(concepts),
        laws=tuple(law_nodes),
        concept_to_law_edges=tuple(dict.fromkeys(concept_to_law_edges)),
        law_to_dim_edges=tuple(dict.fromkeys(law_to_dim_edges)),
        dimension_budget=tuple(float(max(0.0, min(1.0, value))) for value in budget),
    )


def _concept_nodes(
    env_id: str | None,
    state_dim: int,
    action_dim: int,
) -> tuple[ConceptNode, ...]:
    spec = get_mujoco_task_spec(env_id or "unknown", state_dim=state_dim, action_dim=action_dim)
    nodes: dict[str, ConceptNode] = {}
    for dim in spec.state_dimensions:
        base_names = {
            _concept_name(dim.kind),
            _concept_name(dim.name),
            *(_concept_name(token) for token in _name_tokens(dim.name)),
        }
        for name in base_names:
            nodes[name] = ConceptNode(
                name=name,
                kind=dim.kind,
                dimension_indices=(int(dim.index),),
            )
    for action in spec.action_dimensions:
        base_names = {
            _concept_name(action.kind),
            _concept_name(action.name),
            *(_concept_name(token) for token in _name_tokens(action.name)),
        }
        for name in base_names:
            nodes[name] = ConceptNode(
                name=name,
                kind=action.kind,
                action_indices=(int(action.index),),
            )
    nodes["dt"] = ConceptNode(name="dt", kind="time_step")
    return tuple(sorted(nodes.values(), key=lambda node: node.name))


def _dimension_concept_names(
    concepts: Sequence[ConceptNode],
) -> dict[int, tuple[str, ...]]:
    grouped: dict[int, list[str]] = {}
    for concept in concepts:
        for dim in concept.dimension_indices:
            grouped.setdefault(dim, []).append(concept.name)
    return {
        dim: tuple(sorted(set(names)))
        for dim, names in grouped.items()
    }


def _action_concept_names(
    concepts: Sequence[ConceptNode],
) -> tuple[str, ...]:
    names = [
        concept.name
        for concept in concepts
        if concept.action_indices or concept.kind in {"force", "torque", "ctrl"}
    ]
    return tuple(sorted(set(names)))


def _law_target_dims(
    law: ContinuousLawProtocol,
    batch: TransitionBatch,
    validation_sample_count: int,
) -> tuple[int, ...]:
    targets: set[int] = set()
    limit = _validation_limit(batch, validation_sample_count)
    for idx in range(limit):
        state = batch.states[idx]
        action = batch.actions[idx]
        try:
            if not law.precondition(state, action):
                continue
            prediction = law.predict(state, action)
        except Exception:
            continue
        for raw_index in prediction.indices.detach().cpu().to(torch.long).tolist():
            index = int(raw_index)
            if 0 <= index < int(batch.states.shape[1]):
                targets.add(index)
    return tuple(sorted(targets))


def _law_quality(
    law: ContinuousLawProtocol,
    batch: TransitionBatch,
    target_dims: tuple[int, ...],
    transition_dt: float,
    validation_sample_count: int,
    quality_floor: float,
) -> float:
    if not target_dims:
        return 0.0
    limit = _validation_limit(batch, validation_sample_count)
    if limit <= 1:
        return float(quality_floor)

    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    dims = torch.tensor(target_dims, dtype=torch.long, device=batch.states.device)
    for idx in range(limit):
        state = batch.states[idx]
        action = batch.actions[idx]
        next_state = batch.next_states[idx]
        predicted_next = state.clone()
        try:
            if law.precondition(state, action):
                prediction = law.predict(state, action)
                indices = prediction.indices.to(device=state.device, dtype=torch.long)
                values = prediction.values.to(device=state.device, dtype=state.dtype)
                local_delta = _prediction_to_delta(
                    prediction=prediction,
                    state=state,
                    indices=indices,
                    values=values,
                    transition_dt=transition_dt,
                )
                predicted_next[indices] = state[indices] + local_delta
        except Exception:
            pass
        predictions.append(predicted_next[dims])
        targets.append(next_state[dims])
        states.append(state[dims])
    pred = torch.stack(predictions)
    target = torch.stack(targets)
    state_baseline = torch.stack(states)
    r2 = _delta_r2_uniform(
        states=state_baseline,
        predictions=pred,
        next_states=target,
    )
    if not torch.isfinite(torch.tensor(r2)):
        return float(quality_floor)
    positive = max(0.0, float(r2))
    return float(max(quality_floor, min(1.0, positive)))


def _law_parent_concepts(
    law_name: str,
    target_dims: tuple[int, ...],
    dimension_concepts: dict[int, tuple[str, ...]],
    action_concepts: tuple[str, ...],
    concepts_by_name: dict[str, ConceptNode],
) -> tuple[str, ...]:
    tokens = {_concept_name(token) for token in _name_tokens(law_name)}
    parents: set[str] = {"dt"} if "dt" in concepts_by_name else set()
    for dim in target_dims:
        parents.update(dimension_concepts.get(dim, ()))
    for concept in concepts_by_name:
        if concept in tokens:
            parents.add(concept)
    if tokens.intersection({"action", "ctrl", "torque", "force"}):
        parents.update(action_concepts)
    return tuple(sorted(parent for parent in parents if parent in concepts_by_name))


def _delta_r2_uniform(
    states: torch.Tensor,
    predictions: torch.Tensor,
    next_states: torch.Tensor,
) -> float:
    target = next_states - states
    predicted = predictions - states
    residual_sum = (target - predicted).square().sum(dim=0)
    centered = target - target.mean(dim=0, keepdim=True)
    total_sum = centered.square().sum(dim=0)
    valid = total_sum > 1e-12
    if not bool(torch.any(valid)):
        return 0.0
    r2 = 1.0 - residual_sum[valid] / total_sum[valid]
    return float(r2.mean().cpu())


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
    return values


def _validation_limit(
    batch: TransitionBatch,
    validation_sample_count: int,
) -> int:
    return min(int(batch.states.shape[0]), max(1, int(validation_sample_count)))


def _law_name(law: ContinuousLawProtocol, law_index: int) -> str:
    raw = getattr(law, "law_name", None)
    if raw is None:
        raw = getattr(law, "_law_name", None)
    return str(raw or f"law_{law_index}")


def _name_tokens(text: str) -> tuple[str, ...]:
    return tuple(token for token in re.split(r"[^a-zA-Z0-9]+", text.lower()) if token)


def _concept_name(text: str) -> str:
    tokens = _name_tokens(text)
    return "_".join(tokens) if tokens else "unknown"
