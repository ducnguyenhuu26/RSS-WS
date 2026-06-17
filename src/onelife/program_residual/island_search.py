from __future__ import annotations

import copy
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache

import torch
import torch.nn.functional as F

try:
    import gymnasium as gym
except ModuleNotFoundError:  # pragma: no cover - exercised only in slim test envs.
    gym = None

from .core import TransitionBatch
from .laws import ContinuousLaw
from .llm_synthesizer import (
    LLMLawSynthesisConfig,
    LLMSymbolicLawSynthesizer,
    LLMSynthesizedLaws,
)
from .program import SymbolicProgram
from .task_specs import get_mujoco_task_spec


@dataclass(frozen=True)
class NicheSpec:
    name: str
    instructions: str
    coverage_weight: float = 0.05
    complexity_weight: float = 0.01
    focus_weight: float = 0.05


@dataclass(frozen=True)
class IslandSearchConfig:
    env_id: str | None = None
    candidates_per_niche: int = 1
    generations: int = 1
    island_size: int = 4
    elite_per_island: int = 1
    migration_interval: int = 1
    migrants_per_island: int = 1
    max_laws_per_program: int = 4
    validation_sample_count: int = 512


@dataclass(frozen=True)
class CandidateMetrics:
    fitness: float
    delta_r2_uniform: float
    program_mse: float
    identity_mse: float
    coverage: float
    position_coverage: float
    velocity_coverage: float
    num_laws: int
    mean_confidence: float
    worse_than_identity: float


@dataclass(frozen=True)
class LawProgramCandidate:
    laws: tuple[ContinuousLaw, ...]
    code: str
    niche: str
    origin: str
    generation: int
    metrics: CandidateMetrics


@dataclass(frozen=True)
class IslandSearchResult:
    bundle: LLMSynthesizedLaws
    best_candidate: LawProgramCandidate
    candidates: tuple[LawProgramCandidate, ...]
    summary: dict[str, float | int | str]


NICHES: tuple[NicheSpec, ...] = (
    NicheSpec(
        name="kinematic",
        instructions=(
            "Focus on high-precision kinematic laws, especially position or angle "
            "updates of the form q_next = q + dt * qdot. Use the semantic state "
            "labels to find matching qpos/qvel concept leaders. Implement each "
            "law as a follower code node attached to explicit concept parents. "
            "Avoid velocity/contact laws unless the transition samples strongly "
            "justify them."
        ),
        coverage_weight=0.03,
        complexity_weight=0.015,
        focus_weight=0.08,
    ),
    NicheSpec(
        name="action_dynamics",
        instructions=(
            "Focus on action-conditioned velocity or angular-velocity changes. "
            "Prefer sparse linear action effects and damping terms. Treat actions "
            "as ctrl/torque concept leaders and implement only local qvel follower "
            "laws. Do not predict position dimensions unless needed for a "
            "precondition."
        ),
        coverage_weight=0.04,
        complexity_weight=0.012,
        focus_weight=0.08,
    ),
    NicheSpec(
        name="sparse_conservative",
        instructions=(
            "Be conservative. Predict only dimensions that are clearly improved "
            "over identity dynamics. Prefer few laws, narrow dimension masks, and "
            "lower confidence/std-aware probabilistic evidence over broad "
            "speculative rules. Good followers here are identity/near-identity "
            "laws for target, geometry, or constraint concepts."
        ),
        coverage_weight=0.015,
        complexity_weight=0.03,
        focus_weight=0.04,
    ),
    NicheSpec(
        name="broad_exploratory",
        instructions=(
            "Explore broader but still interpretable laws. It is acceptable to "
            "cover more dimensions, but each law must remain safe, executable, "
            "and grounded in the observed transition deltas. Keep the law set as "
            "a DAG of small follower nodes rather than one monolithic program."
        ),
        coverage_weight=0.08,
        complexity_weight=0.006,
        focus_weight=0.02,
    ),
)


def synthesize_with_island_search(
    synthesizer: LLMSymbolicLawSynthesizer,
    batch: TransitionBatch,
    config: LLMLawSynthesisConfig,
    search_config: IslandSearchConfig,
) -> IslandSearchResult:
    initial = _generate_initial_population(synthesizer, batch, config, search_config)
    if not initial:
        empty_metrics = evaluate_laws((), batch, NICHES[2], search_config)
        empty = LawProgramCandidate(
            laws=(),
            code="def build_laws(state_dim, action_dim, dt, confidence):\n    return []",
            niche="sparse_conservative",
            origin="empty_fallback",
            generation=0,
            metrics=empty_metrics,
        )
        return _result_from_best(empty, (empty,), config, batch)

    islands = _partition_into_niches(initial, search_config)
    all_candidates = list(initial)
    for generation in range(1, max(0, search_config.generations) + 1):
        islands = {
            niche: _evolve_island(
                niche=niche,
                population=population,
                batch=batch,
                search_config=search_config,
                generation=generation,
            )
            for niche, population in islands.items()
        }
        if (
            search_config.migration_interval > 0
            and generation % search_config.migration_interval == 0
        ):
            islands = _migrate_ring(islands, search_config)
        for population in islands.values():
            all_candidates.extend(population)

    final_population = [
        candidate for population in islands.values() for candidate in population
    ]
    unique_candidates = _unique_candidates([*all_candidates, *final_population])
    archive_union = _make_archive_union_candidate(
        unique_candidates,
        batch,
        search_config,
    )
    if archive_union is not None:
        unique_candidates.append(archive_union)
    best = max(unique_candidates or initial, key=lambda candidate: candidate.metrics.fitness)
    return _result_from_best(best, tuple(unique_candidates), config, batch)


def _generate_initial_population(
    synthesizer: LLMSymbolicLawSynthesizer,
    batch: TransitionBatch,
    config: LLMLawSynthesisConfig,
    search_config: IslandSearchConfig,
) -> list[LawProgramCandidate]:
    candidates: list[LawProgramCandidate] = []
    for niche in NICHES:
        for candidate_idx in range(max(1, search_config.candidates_per_niche)):
            niche_config = replace(
                config,
                niche=niche.name,
                extra_instructions=(
                    f"{niche.instructions}\n"
                    f"Candidate id within this niche: {candidate_idx}. Prefer a "
                    "distinct law set from other candidates. Name each law with "
                    "its leader/follower semantic parent concepts, for example "
                    "'follower__qpos_cart_position__qvel_cart_velocity' or "
                    "'follower__qvel_joint_velocity__ctrl_joint_torque'."
                ),
            )
            try:
                bundle = synthesizer.synthesize_from_batch(batch, niche_config)
            except Exception as exc:
                warnings.warn(
                    "Skipping failed island-search LLM candidate "
                    f"{niche.name}:{candidate_idx}: {type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            assigned_niche = classify_laws(bundle.laws, batch, search_config)
            assigned_spec = _niche_by_name(assigned_niche)
            candidates.append(
                LawProgramCandidate(
                    laws=tuple(bundle.laws),
                    code=bundle.code,
                    niche=assigned_niche,
                    origin=f"llm:{niche.name}:{candidate_idx}",
                    generation=0,
                    metrics=evaluate_laws(
                        bundle.laws,
                        batch,
                        assigned_spec,
                        search_config,
                    ),
                )
            )
    return candidates


def _partition_into_niches(
    candidates: Sequence[LawProgramCandidate],
    search_config: IslandSearchConfig,
) -> dict[str, list[LawProgramCandidate]]:
    islands = {niche.name: [] for niche in NICHES}
    for candidate in candidates:
        islands.setdefault(candidate.niche, []).append(candidate)
    for niche in NICHES:
        population = _select_survivors(
            islands[niche.name],
            search_config.island_size,
            niche,
        )
        if len(population) < search_config.island_size:
            donors = sorted(
                candidates,
                key=lambda candidate: _niche_affinity(candidate.metrics, niche),
                reverse=True,
            )
            for donor in donors:
                if len(population) >= search_config.island_size:
                    break
                population.append(_retarget_candidate(donor, niche))
        islands[niche.name] = _select_survivors(
            population,
            search_config.island_size,
            niche,
        )
    return islands


def _evolve_island(
    niche: str,
    population: list[LawProgramCandidate],
    batch: TransitionBatch,
    search_config: IslandSearchConfig,
    generation: int,
) -> list[LawProgramCandidate]:
    spec = _niche_by_name(niche)
    ranked = _select_survivors(
        population,
        max(1, min(len(population), search_config.elite_per_island + 3)),
        spec,
    )
    elites = ranked[: max(1, search_config.elite_per_island)]
    children: list[LawProgramCandidate] = []
    if len(ranked) >= 2:
        children.append(
            _crossover_candidates(
                ranked[0],
                ranked[1],
                spec,
                batch,
                search_config,
                generation,
            )
        )
    for elite in elites:
        children.append(
            _mutate_candidate(
                elite,
                spec,
                generation,
                batch,
                search_config,
            )
        )
    combined = elites + children + ranked[len(elites) :] + population
    return _select_survivors(combined, search_config.island_size, spec)


def _migrate_ring(
    islands: dict[str, list[LawProgramCandidate]],
    search_config: IslandSearchConfig,
) -> dict[str, list[LawProgramCandidate]]:
    names = [niche.name for niche in NICHES]
    migrants: dict[str, list[LawProgramCandidate]] = {}
    for name in names:
        population = sorted(
            islands.get(name, []),
            key=lambda candidate: candidate.metrics.fitness,
            reverse=True,
        )
        migrants[name] = population[: max(0, search_config.migrants_per_island)]
    updated = {name: list(islands.get(name, [])) for name in names}
    for idx, name in enumerate(names):
        target = names[(idx + 1) % len(names)]
        target_spec = _niche_by_name(target)
        for migrant in migrants[name]:
            adapted = LawProgramCandidate(
                laws=tuple(_clone_law(law) for law in migrant.laws),
                code=migrant.code,
                niche=target,
                origin=f"migrate:{name}->{target}",
                generation=migrant.generation,
                metrics=_retarget_fitness(migrant.metrics, target_spec),
            )
            updated[target].append(adapted)
    return {
        name: _select_survivors(
            population,
            search_config.island_size,
            _niche_by_name(name),
        )
        for name, population in updated.items()
    }


def classify_laws(
    laws: Sequence[ContinuousLaw],
    batch: TransitionBatch,
    search_config: IslandSearchConfig,
) -> str:
    metrics_by_niche = {
        niche.name: evaluate_laws(laws, batch, niche, search_config)
        for niche in NICHES
    }
    coverage = metrics_by_niche["broad_exploratory"].coverage
    num_laws = metrics_by_niche["broad_exploratory"].num_laws
    if num_laws <= 1 or coverage < 0.25:
        return "sparse_conservative"
    if metrics_by_niche["kinematic"].position_coverage >= max(
        0.05,
        metrics_by_niche["kinematic"].velocity_coverage,
    ):
        return "kinematic"
    if metrics_by_niche["action_dynamics"].velocity_coverage >= 0.05:
        return "action_dynamics"
    return "broad_exploratory"


def evaluate_laws(
    laws: Sequence[ContinuousLaw],
    batch: TransitionBatch,
    niche: NicheSpec,
    search_config: IslandSearchConfig,
) -> CandidateMetrics:
    limit = min(search_config.validation_sample_count, int(batch.states.shape[0]))
    if limit <= 0:
        raise ValueError("cannot evaluate island candidate on empty batch")
    eval_batch = TransitionBatch(
        states=batch.states[:limit],
        actions=batch.actions[:limit],
        next_states=batch.next_states[:limit],
    )
    state_dim = int(eval_batch.states.shape[1])
    action_dim = int(eval_batch.actions.shape[1])
    program = SymbolicProgram(state_dim=state_dim, laws=[_clone_law(law) for law in laws])
    with torch.no_grad():
        output = program(eval_batch.states, eval_batch.actions)
        program_mse = float(F.mse_loss(output.next_state, eval_batch.next_states).cpu())
        identity_mse = float(F.mse_loss(eval_batch.states, eval_batch.next_states).cpu())
        delta_r2 = _delta_r2_uniform(
            states=eval_batch.states,
            predictions=output.next_state,
            next_states=eval_batch.next_states,
        )
        known_mask = 1.0 - output.unknown_mask.float()
        coverage = float(known_mask.mean().cpu())
        position_indices, velocity_indices = _dimension_groups(
            search_config.env_id,
            state_dim,
        )
        position_coverage = (
            float(known_mask[:, list(position_indices)].mean().cpu())
            if position_indices
            else 0.0
        )
        velocity_coverage = (
            float(known_mask[:, list(velocity_indices)].mean().cpu())
            if velocity_indices
            else 0.0
        )
        mean_confidence = float(output.confidence.float().mean().cpu())
    worse_than_identity = max(0.0, program_mse - identity_mse) / max(identity_mse, 1e-8)
    if niche.name == "kinematic":
        focus = position_coverage
    elif niche.name == "action_dynamics":
        focus = velocity_coverage
    elif niche.name == "sparse_conservative":
        focus = 1.0 / max(1, len(laws))
    else:
        focus = coverage
    fitness = (
        delta_r2
        + niche.coverage_weight * coverage
        + niche.focus_weight * focus
        - niche.complexity_weight * len(laws)
        - worse_than_identity
    )
    return CandidateMetrics(
        fitness=float(fitness),
        delta_r2_uniform=float(delta_r2),
        program_mse=program_mse,
        identity_mse=identity_mse,
        coverage=coverage,
        position_coverage=position_coverage,
        velocity_coverage=velocity_coverage,
        num_laws=len(laws),
        mean_confidence=mean_confidence,
        worse_than_identity=float(worse_than_identity),
    )


def _crossover_candidates(
    parent_a: LawProgramCandidate,
    parent_b: LawProgramCandidate,
    niche: NicheSpec,
    batch: TransitionBatch,
    search_config: IslandSearchConfig,
    generation: int,
) -> LawProgramCandidate:
    law_pool = [_clone_law(law) for law in (*parent_a.laws, *parent_b.laws)]
    child_laws = _select_semantic_crossover_laws(
        law_pool=law_pool,
        niche=niche,
        batch=batch,
        search_config=search_config,
        max_laws=max(1, search_config.max_laws_per_program),
    )
    return LawProgramCandidate(
        laws=child_laws,
        code=f"# crossover({parent_a.origin}, {parent_b.origin})",
        niche=niche.name,
        origin=f"crossover:{parent_a.origin}+{parent_b.origin}",
        generation=generation,
        metrics=evaluate_laws(child_laws, batch, niche, search_config),
    )


def _mutate_candidate(
    candidate: LawProgramCandidate,
    niche: NicheSpec,
    generation: int,
    batch: TransitionBatch | None,
    search_config: IslandSearchConfig,
) -> LawProgramCandidate:
    laws = [_clone_law(law) for law in candidate.laws]
    if batch is not None and len(laws) > 1:
        worst_index = _semantic_mutation_drop_index(laws, batch, niche, search_config)
        laws = [law for index, law in enumerate(laws) if index != worst_index]
    elif len(laws) > search_config.max_laws_per_program:
        laws = laws[: search_config.max_laws_per_program]
    elif laws and niche.name == "sparse_conservative":
        laws = [_attenuate_law_confidence(law, factor=0.5) for law in laws]
    if batch is None:
        metrics = _retarget_fitness(candidate.metrics, niche)
    else:
        metrics = evaluate_laws(laws, batch, niche, search_config)
    return LawProgramCandidate(
        laws=tuple(laws),
        code=f"# mutate({candidate.origin})",
        niche=niche.name,
        origin=f"mutate:{candidate.origin}",
        generation=generation,
        metrics=metrics,
    )


def _select_semantic_crossover_laws(
    law_pool: Sequence[ContinuousLaw],
    niche: NicheSpec,
    batch: TransitionBatch,
    search_config: IslandSearchConfig,
    max_laws: int,
) -> tuple[ContinuousLaw, ...]:
    if not law_pool or max_laws <= 0:
        return ()
    scored = [
        (
            law,
            evaluate_laws((law,), batch, niche, search_config).fitness,
            _law_semantic_tokens(law, batch, search_config),
        )
        for law in law_pool
    ]
    selected: list[tuple[ContinuousLaw, float, frozenset[str]]] = []
    remaining = list(scored)
    semantic_novelty_weight = 0.20 if niche.name == "broad_exploratory" else 0.12
    while remaining and len(selected) < max_laws:
        if not selected:
            chosen = max(remaining, key=lambda item: item[1])
        else:
            chosen = max(
                remaining,
                key=lambda item: item[1]
                + semantic_novelty_weight
                * _semantic_novelty(item[2], [existing[2] for existing in selected]),
            )
        selected.append(chosen)
        remaining.remove(chosen)
    return tuple(_clone_law(item[0]) for item in selected)


def _semantic_mutation_drop_index(
    laws: Sequence[ContinuousLaw],
    batch: TransitionBatch,
    niche: NicheSpec,
    search_config: IslandSearchConfig,
) -> int:
    profiles = [_law_semantic_tokens(law, batch, search_config) for law in laws]
    scores: list[tuple[float, int]] = []
    for index, law in enumerate(laws):
        fitness = evaluate_laws((law,), batch, niche, search_config).fitness
        redundancy = max(
            (
                _semantic_similarity(profiles[index], other)
                for other_idx, other in enumerate(profiles)
                if other_idx != index
            ),
            default=0.0,
        )
        scores.append((fitness - 0.20 * redundancy, index))
    _score, index_to_drop = min(scores, key=lambda item: item[0])
    return index_to_drop


def _law_semantic_tokens(
    law: ContinuousLaw,
    batch: TransitionBatch,
    search_config: IslandSearchConfig,
) -> frozenset[str]:
    tokens = set(_name_tokens(_law_signature(law)))
    state_dim = int(batch.states.shape[1])
    concepts = _dimension_concepts(search_config.env_id, state_dim)
    limit = min(int(batch.states.shape[0]), max(8, search_config.validation_sample_count // 8))
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
            tokens.add(f"state_{index}")
            tokens.update(concepts.get(index, ()))
    return frozenset(tokens)


def _dimension_concepts(
    env_id: str | None,
    state_dim: int,
) -> dict[int, tuple[str, ...]]:
    concepts: dict[int, tuple[str, ...]] = {}
    if env_id:
        spec = get_mujoco_task_spec(env_id, state_dim=state_dim, action_dim=None)
        for dim in spec.state_dimensions:
            if dim.kind == "unknown":
                continue
            concepts[dim.index] = tuple(
                sorted(
                    {
                        dim.kind.lower(),
                        dim.name.lower(),
                        *_name_tokens(dim.name),
                    }
                )
            )
    if concepts:
        return concepts
    position_indices, velocity_indices = _dimension_groups(env_id, state_dim)
    for index in position_indices:
        concepts[index] = ("qpos", f"state_{index}")
    for index in velocity_indices:
        concepts[index] = ("qvel", f"state_{index}")
    return concepts


def _semantic_novelty(
    tokens: frozenset[str],
    existing: Sequence[frozenset[str]],
) -> float:
    if not existing:
        return 1.0
    return 1.0 - max(_semantic_similarity(tokens, other) for other in existing)


def _semantic_similarity(
    left: frozenset[str],
    right: frozenset[str],
) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _name_tokens(value: str) -> tuple[str, ...]:
    token = []
    tokens: list[str] = []
    for char in value.lower():
        if char.isalnum():
            token.append(char)
        elif token:
            tokens.append("".join(token))
            token = []
    if token:
        tokens.append("".join(token))
    return tuple(tokens)


def _retarget_candidate(
    candidate: LawProgramCandidate,
    niche: NicheSpec,
) -> LawProgramCandidate:
    return LawProgramCandidate(
        laws=tuple(_clone_law(law) for law in candidate.laws),
        code=candidate.code,
        niche=niche.name,
        origin=f"retarget:{candidate.origin}->{niche.name}",
        generation=candidate.generation,
        metrics=_retarget_fitness(candidate.metrics, niche),
    )


def _select_survivors(
    population: Sequence[LawProgramCandidate],
    size: int,
    niche: NicheSpec,
) -> list[LawProgramCandidate]:
    if size <= 0 or not population:
        return []
    selected: list[LawProgramCandidate] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()

    def add(candidate: LawProgramCandidate) -> None:
        if len(selected) >= size:
            return
        signature = _candidate_signature(candidate)
        if signature in seen:
            return
        seen.add(signature)
        selected.append(candidate)

    ranking_functions = [
        lambda item: item.metrics.fitness,
        lambda item: item.metrics.delta_r2_uniform,
        lambda item: item.metrics.coverage,
        lambda item: -item.metrics.num_laws,
    ]
    if niche.name == "kinematic":
        ranking_functions.append(lambda item: item.metrics.position_coverage)
    elif niche.name == "action_dynamics":
        ranking_functions.append(lambda item: item.metrics.velocity_coverage)
    elif niche.name == "sparse_conservative":
        ranking_functions.append(lambda item: -item.metrics.worse_than_identity)
    else:
        ranking_functions.append(lambda item: item.metrics.position_coverage + item.metrics.velocity_coverage)

    for ranking in ranking_functions:
        for candidate in sorted(population, key=ranking, reverse=True):
            previous_count = len(selected)
            add(candidate)
            if len(selected) > previous_count:
                break

    for candidate in sorted(
        population,
        key=lambda item: item.metrics.fitness,
        reverse=True,
    ):
        add(candidate)
    return selected


def _make_archive_union_candidate(
    candidates: Sequence[LawProgramCandidate],
    batch: TransitionBatch,
    search_config: IslandSearchConfig,
) -> LawProgramCandidate | None:
    law_pool: list[ContinuousLaw] = []
    for candidate in sorted(
        candidates,
        key=lambda item: item.metrics.fitness,
        reverse=True,
    ):
        for law in candidate.laws:
            law_pool.append(_clone_law(law))
    if not law_pool:
        return None
    broad = _niche_by_name("broad_exploratory")
    ranked_laws = sorted(
        law_pool,
        key=lambda law: evaluate_laws((law,), batch, broad, search_config).fitness,
        reverse=True,
    )
    selected: list[ContinuousLaw] = []
    seen: set[str] = set()
    for law in ranked_laws:
        signature = _law_signature(law)
        if signature in seen:
            continue
        seen.add(signature)
        selected.append(law)
        if len(selected) >= search_config.max_laws_per_program:
            break
    if not selected:
        return None
    niche_name = classify_laws(selected, batch, search_config)
    niche = _niche_by_name(niche_name)
    return LawProgramCandidate(
        laws=tuple(selected),
        code="# archive_union",
        niche=niche.name,
        origin="archive_union",
        generation=search_config.generations,
        metrics=evaluate_laws(selected, batch, niche, search_config),
    )


def _retarget_fitness(metrics: CandidateMetrics, niche: NicheSpec) -> CandidateMetrics:
    if niche.name == "kinematic":
        focus = metrics.position_coverage
    elif niche.name == "action_dynamics":
        focus = metrics.velocity_coverage
    elif niche.name == "sparse_conservative":
        focus = 1.0 / max(1, metrics.num_laws)
    else:
        focus = metrics.coverage
    fitness = (
        metrics.delta_r2_uniform
        + niche.coverage_weight * metrics.coverage
        + niche.focus_weight * focus
        - niche.complexity_weight * metrics.num_laws
        - metrics.worse_than_identity
    )
    return replace(metrics, fitness=float(fitness))


def _niche_affinity(metrics: CandidateMetrics, niche: NicheSpec) -> float:
    return _retarget_fitness(metrics, niche).fitness


def _niche_by_name(name: str) -> NicheSpec:
    for niche in NICHES:
        if niche.name == name:
            return niche
    return NICHES[-1]


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


@lru_cache(maxsize=64)
def _dimension_groups(
    env_id: str | None,
    state_dim: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if env_id:
        groups = _dimension_groups_from_task_spec(env_id, state_dim)
        if groups is not None:
            return groups
        groups = _dimension_groups_from_gym(env_id, state_dim)
        if groups is not None:
            return groups
    split = max(0, state_dim // 2)
    return tuple(range(split)), tuple(range(split, state_dim))


def _dimension_groups_from_task_spec(
    env_id: str,
    state_dim: int,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    spec = get_mujoco_task_spec(env_id, state_dim=state_dim, action_dim=None)
    position_indices: list[int] = []
    velocity_indices: list[int] = []
    for dim in spec.state_dimensions:
        kind = dim.kind.lower()
        if kind == "unknown":
            continue
        if "qvel" in kind or "velocity" in kind:
            velocity_indices.append(dim.index)
        elif "qpos" in kind or kind in {"sin_angle", "cos_angle"}:
            position_indices.append(dim.index)
    if not position_indices and not velocity_indices:
        return None
    return tuple(position_indices), tuple(velocity_indices)


def _dimension_groups_from_gym(
    env_id: str,
    state_dim: int,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    if gym is None:
        return None
    try:
        env = gym.make(env_id)
    except Exception:
        return None
    try:
        structure = getattr(env.unwrapped, "observation_structure", None)
    finally:
        env.close()
    if not isinstance(structure, dict):
        return None

    position_indices: list[int] = []
    velocity_indices: list[int] = []
    offset = 0
    for raw_name, raw_count in structure.items():
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            return None
        indices = [
            index
            for index in range(offset, min(offset + count, state_dim))
            if index >= 0
        ]
        offset += count
        name = str(raw_name).lower()
        if name.startswith("skipped"):
            continue
        if "qvel" in name or "velocity" in name or name.endswith("vel"):
            velocity_indices.extend(indices)
        else:
            position_indices.extend(indices)

    if not position_indices and not velocity_indices:
        return None
    return tuple(position_indices), tuple(velocity_indices)


def _clone_law(law: ContinuousLaw) -> ContinuousLaw:
    return copy.deepcopy(law)


def _attenuate_law_confidence(
    law: ContinuousLaw,
    factor: float,
) -> ContinuousLaw:
    mutated = _clone_law(law)
    if hasattr(mutated, "confidence_value"):
        current = getattr(mutated, "confidence_value")
        try:
            setattr(mutated, "confidence_value", float(current) * float(factor))
        except (TypeError, ValueError):
            pass
    return mutated


def _law_signature(law: ContinuousLaw) -> str:
    return f"{law.__class__.__name__}:{getattr(law, 'law_name', law.__class__.__name__)}"


def _candidate_signature(
    candidate: LawProgramCandidate,
) -> tuple[str, str, tuple[str, ...]]:
    return (
        candidate.niche,
        candidate.origin,
        tuple(_law_signature(law) for law in candidate.laws),
    )


def _unique_candidates(
    candidates: Sequence[LawProgramCandidate],
) -> list[LawProgramCandidate]:
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    unique: list[LawProgramCandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: item.metrics.fitness,
        reverse=True,
    ):
        key = _candidate_signature(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _result_from_best(
    best: LawProgramCandidate,
    candidates: tuple[LawProgramCandidate, ...],
    config: LLMLawSynthesisConfig,
    batch: TransitionBatch,
) -> IslandSearchResult:
    code = "\n\n".join(
        candidate.code
        for candidate in candidates
        if candidate.origin.startswith("llm:")
    )
    bundle = LLMSynthesizedLaws(
        laws=tuple(_clone_law(law) for law in best.laws),
        code=code or best.code,
        prompt=f"island_search:{config.env_id}",
        raw_response="",
    )
    summary = {
        "best_fitness": best.metrics.fitness,
        "best_delta_r2_uniform": best.metrics.delta_r2_uniform,
        "best_coverage": best.metrics.coverage,
        "best_num_laws": best.metrics.num_laws,
        "best_niche": best.niche,
        "best_origin": best.origin,
        "num_candidates": len(candidates),
        "state_dim": int(batch.states.shape[1]),
    }
    return IslandSearchResult(
        bundle=bundle,
        best_candidate=best,
        candidates=candidates,
        summary=summary,
    )
