# MuJoCo-Adaptive OneLife World Modeling

This repository is based on One Life to Learn and adapts the symbolic OneLife
world-modeling framework to MuJoCo continuous-control environments. The current
experiments compare adaptive OneLife against five variants of our MuJoCo
framework.

## Final Evaluation

Each environment is reported with two columns:

- **Score**: `one_step_delta_r2_uniform`, the mean per-state-dimension R2 for
  predicting the one-step state delta `s_next - s`.
- **Reward**: planner return in the real environment using the learned world
  model, reported as `Random MPC / CEM-MPC`.

The primary score is delta R2 rather than next-state R2 because identity
prediction can look strong on MuJoCo next states while failing to learn the
actual transition dynamics.

The planner reward is actual environment reward accumulated after executing
actions chosen by MPC. The model is only used inside the planner to score
candidate action sequences.

## Environments

Use the MuJoCo v5 tasks up to HalfCheetah:

```text
InvertedPendulum-v5
InvertedDoublePendulum-v5
Reacher-v5
Hopper-v5
Walker2d-v5
HalfCheetah-v5
```

Do not include:

```text
Ant-v5
Pusher-v5
```

## Model Set

The final table keeps the previous comparison principle: adaptive OneLife plus
five variants of our framework.

```text
onelife          Adaptive OneLife binned LLM law mixture
ours             LLM symbolic program + neural residual
program_only     LLM symbolic program only
neural           Neural residual only
symbolic         Standard symbolic prior only
symbolic_neural  Standard symbolic prior + neural residual
```

## Running

Default Hydra config is in `configs/config.yaml`. The smoke config is
`configs/smoke.yaml`. `device: auto` uses CUDA when PyTorch can see a GPU and
falls back to CPU otherwise.

Check the GPU environment on the 3060 machine before the sweep:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

```bash
uv run --env-file .env python main.py --config-name smoke
```

Example full sweep command:

```bash
uv run --env-file .env python main.py -m \
  problem=InvertedPendulum-v5,InvertedDoublePendulum-v5,Reacher-v5,Hopper-v5,Walker2d-v5,HalfCheetah-v5 \
  model=onelife,ours,program_only,neural,symbolic,symbolic_neural \
  seed=0,1,2
```

Format the final markdown tables from JSON outputs:

```bash
uv run --env-file .env python scripts/format_mujoco_final_table.py outputs/*.json
```

Aggregate one metric across seeds:

```bash
uv run --env-file .env python scripts/aggregate_mujoco_results.py outputs/*.json \
  --metric score.one_step_delta_r2_uniform --show-std
```

LLM call counts are saved in every output JSON as top-level `llm_calls` and a
larger `llm_usage` object. Aggregate them with:

```bash
uv run --env-file .env python scripts/aggregate_mujoco_results.py outputs/*.json \
  --metric llm_calls --show-std
```

## Notes

- Continuous `ours` variants use `ProgramResidualWorldModel`.
- Adaptive OneLife uses binned MuJoCo states/actions and OneLife-style
  precondition-effect laws.
- The neural residual is a full correction term, not masked to only unknown
  symbolic dimensions.
- Planner scoring uses a task proxy inside MPC; reported Reward is real
  environment return after executing the selected actions.
- Each LLM-based run records logical synthesis calls and available token usage
  for ablations.
