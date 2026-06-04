# Final Experiment Version

This version changes the MuJoCo-adaptive OneLife experiments to report both
prediction quality and planning utility.

## Main Table

Each environment should be shown with two columns:

```text
Score   = one_step_delta_r2_uniform
Reward  = Random MPC return / CEM-MPC return
```

`Score` is R2 on one-step state deltas. It is computed per state dimension and
then averaged uniformly across dimensions:

```text
target_delta = s_next - s
pred_delta   = model(s, a) - s
```

This avoids giving too much credit to identity prediction on slowly changing
MuJoCo states.

`Reward` is actual environment return from model-predictive control. The learned
world model is used by the planner to choose actions; the selected actions are
executed in the real Gymnasium environment.

## Planners

Two planners are included:

```text
Random MPC
CEM-MPC
```

Both use the same learned model interface. Random MPC samples action sequences
uniformly. CEM-MPC iteratively refits a Gaussian action-sequence distribution to
elite candidates.

## Compared Models

The final comparison keeps adaptive OneLife plus framework variants including
gated and niche-island hybrids:

```text
onelife
ours_new
ours_gated
ours
program_only
neural
symbolic
symbolic_neural
```

## Environments

Run:

```text
InvertedPendulum-v5
InvertedDoublePendulum-v5
Reacher-v5
Swimmer-v5
Hopper-v5
Walker2d-v5
HalfCheetah-v5
```

Exclude:

```text
Ant-v5
Pusher-v5
```

## Implementation Notes

- `score` and `reward` are now top-level JSON fields.
- `device: auto` selects CUDA for neural training when PyTorch reports an
  available GPU. The resolved device is stored in the output JSON.
- `llm_calls` and `llm_usage` are stored in each output JSON for LLM-call
  ablations. Non-LLM baselines record zero calls.
- `ours_new` performs single-LLM, niche-based symbolic law search:
  kinematic, action-dynamics, sparse-conservative, and broad-exploratory
  islands exchange validated candidates through controlled migration before the
  selected program is passed to the learned gate.
- `ours_gated_island` remains as a compatibility alias for `ours_new`, but new
  result files should use `model=ours_new`.
- Island selection is intentionally soft: each island keeps elites under
  multiple criteria rather than only top fitness, including delta R2, coverage,
  sparsity, and niche-specific focus. A global archive is retained and final
  selection also considers an archive-union program assembled from validated
  laws, reducing the chance that useful laws are discarded early.
- Legacy metrics are still kept under `program_residual`, `onelife_llm`, and
  `symbolic_baselines` for debugging.
- The default aggregation metric is now
  `score.one_step_delta_r2_uniform`.
- `scripts/format_mujoco_final_table.py` formats final markdown tables directly
  from output JSON files.
