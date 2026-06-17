# Final Experiment Version

This version changes the MuJoCo-adaptive OneLife experiments to report both
prediction quality and planning utility.

## Main Table

Each environment should be shown with two columns:

```text
Score   = one_step_delta_r2_uniform
Reward  = CEM-MPC return / PEC-CEM-MPC return
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

The final protocol averages both prediction and planner metrics over three
seeds. Planner evaluation uses one episode per seed.

## Planners

Two planners are included:

```text
CEM-MPC
PEC-CEM-MPC
```

Both use the same learned model interface. CEM-MPC iteratively refits a Gaussian
action-sequence distribution to elite candidates. PEC-CEM-MPC uses the same CEM
optimizer but subtracts planning error/control penalties such as state/action
OOD, symbolic-neural disagreement, and ensemble variance when those signals are
available.

## Compared Models

The main comparison uses four models:

```text
answer
onelife
pets_ensemble
dreamer_v3
```

The ablation comparison uses three models:

```text
neural
program_only
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
- `answer` is the final proposed model: semantic LLM symbolic laws plus
  leader/follower concept prompting, semantic island evolution, weighted-product
  symbolic effect composition, sparse learnable law weights, a probabilistic
  covariance head, and an ODE neural residual.
- `neural` is the architecture-matched ablation: the same neural ODE residual
  and covariance head as `answer`, but with an empty symbolic program.
- LLM-generated laws are soft probabilistic effects over state deltas rather
  than trusted transitions. Their weighted likelihood contributions are trained
  with an L1 penalty, allowing bad symbolic laws to be suppressed by data.
- `dreamer_v3` is an external baseline. Import its outputs as JSON files with
  `model: "dreamer_v3"` and the same final `score` / `reward` schema.
- `pets_ensemble` is a PETS-style bootstrap ensemble of neural dynamics models
  trained on the same offline transitions and evaluated with the same MPC
  planners.
- Continuous MPC scoring includes a risk guard that penalizes OOD predicted
  states/actions, symbolic-neural disagreement, and ensemble variance.
- Island selection is intentionally soft: each island keeps elites under
  multiple criteria rather than only top fitness, including delta R2, coverage,
  sparsity, and niche-specific focus. A global archive is retained and final
  selection also considers an archive-union program assembled from validated
  laws, reducing the chance that useful laws are discarded early.
- Legacy metrics are still kept under `program_residual`, `onelife_llm`, and
  `symbolic_baselines` for debugging when diagnostics are enabled. The final
  config disables extra diagnostics by default to keep the 3-seed full sweep
  tractable.
- The default aggregation metric is now
  `score.one_step_delta_r2_uniform`.
- `scripts/format_mujoco_final_table.py` formats final markdown tables directly
  from output JSON files.
