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

The final comparison keeps adaptive OneLife plus five framework variants:

```text
onelife
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
- Legacy metrics are still kept under `program_residual`, `onelife_llm`, and
  `symbolic_baselines` for debugging.
- The default aggregation metric is now
  `score.one_step_delta_r2_uniform`.
- `scripts/format_mujoco_final_table.py` formats final markdown tables directly
  from output JSON files.
