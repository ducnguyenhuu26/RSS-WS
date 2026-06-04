# MuJoCo-Adaptive OneLife World Modeling

This repository is based on One Life to Learn and adapts the symbolic OneLife
world-modeling framework to MuJoCo continuous-control environments. The current
experiments compare adaptive OneLife and a PETS-style ensemble baseline against
gated, ungated, and niche-island variants of our MuJoCo framework.

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
Final runs average over three seeds. Planner evaluation uses one episode per
seed to keep the full seven-environment sweep tractable.

## Environments

Use the MuJoCo v5 tasks up to HalfCheetah:

```text
InvertedPendulum-v5
InvertedDoublePendulum-v5
Reacher-v5
Swimmer-v5
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

The final table uses seven models: two external baselines, one neural control,
and four variants of the proposed framework. The island variant treats LLM laws
as a population search problem rather than trusting one generated law set.

```text
onelife          Adaptive OneLife binned LLM law mixture
pets_ensemble   PETS-style neural ensemble + MPC
neural          Neural dynamics + MPC
program_only    LLM symbolic program only
ours            LLM symbolic program + neural residual
ours_gated      Gated LLM symbolic program + neural residual
ours_new        Final niche-island LLM law search + gated neural residual
```

## Running

Default Hydra config is in `configs/config.yaml`. The smoke config is
`configs/smoke.yaml`. `device: auto` uses CUDA when PyTorch can see a GPU and
falls back to CPU otherwise.
Use `uv` with `pyproject.toml` and `uv.lock` for setup; stale pip requirements
files are not part of the final MuJoCo run path.

Recommended Windows GPU setup:

```powershell
uv sync --python 3.12
uv pip uninstall torch torchvision torchaudio
uv pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

After installing the CUDA wheel, use `uv run --no-sync ...` for experiment
commands. Running without `--no-sync` may let `uv` restore a CPU-only torch from
the lockfile.

Check the GPU environment before the sweep:

```powershell
uv run --no-sync python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Smoke test:

```powershell
uv run --no-sync --env-file .env python main.py --config-name smoke
```

Example full sweep command:

```powershell
uv run --no-sync --env-file .env python main.py -m problem=InvertedPendulum-v5,InvertedDoublePendulum-v5,Reacher-v5,Swimmer-v5,Hopper-v5,Walker2d-v5,HalfCheetah-v5 model=onelife,pets_ensemble,neural,program_only,ours,ours_gated,ours_new seed=0,1,2 device=cuda skip_existing=true output_dir=outputs_final_3seed
```

Format the final markdown tables from JSON outputs:

```powershell
uv run --no-sync --env-file .env python scripts/format_mujoco_final_table.py outputs_final_3seed/*.json --no-std
```

Aggregate one metric across seeds:

```powershell
uv run --no-sync --env-file .env python scripts/aggregate_mujoco_results.py outputs_final_3seed/*.json `
  --metric score.one_step_delta_r2_uniform --show-std
```

LLM call counts are saved in every output JSON as top-level `llm_calls` and a
larger `llm_usage` object. Aggregate them with:

```powershell
uv run --no-sync --env-file .env python scripts/aggregate_mujoco_results.py outputs_final_3seed/*.json `
  --metric llm_calls --show-std
```

## Notes

- Continuous `ours` variants use `ProgramResidualWorldModel`.
- Adaptive OneLife uses binned MuJoCo states/actions and OneLife-style
  precondition-effect laws.
- The neural residual is a full correction term, not masked to only unknown
  symbolic dimensions.
- `ours_gated` uses a learned dimension-wise gate initialized near neural-only
  behavior. The gate can open on dimensions where symbolic dynamics are useful
  and stays closed on dimensions that symbolic laws do not cover.
- `ours_new` asks a single LLM to propose niche-specific law programs,
  partitions candidates into dynamics-specialized islands, applies numeric
  selection/crossover/mutation/migration, and trains the same gated hybrid on
  the selected program. The search keeps a global archive and uses soft
  multi-objective survivor selection so useful laws are not discarded solely
  because one scalar fitness is temporarily low.
- `ours_gated_island` remains accepted as a compatibility alias, but the paper
  runs should use `ours_new`.
- `pets_ensemble` is a PETS-style baseline: a bootstrap ensemble of neural
  dynamics models trained on the same transitions and evaluated with the same
  MPC planner.
- The final config disables open-loop and symbolic diagnostic metrics by
  default to keep the full 7 environment x 7 model x 3 seed sweep practical.
- Planner scoring uses a task proxy inside MPC; reported Reward is real
  environment return after executing the selected actions.
- Each LLM-based run records logical synthesis calls and available token usage
  for ablations.
