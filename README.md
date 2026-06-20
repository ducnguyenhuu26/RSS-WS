# ANSWER MuJoCo World Modeling

This repository implements **ANSWER**: an automated neuro-symbolic world model
for MuJoCo continuous-control tasks. The code is derived from One Life to Learn,
but the final MuJoCo protocol is intentionally narrow and reproducible: one
proposed method, external/main baselines, and architecture ablations in one
shared table.

## Method

For an environment \(e\), the code first builds a semantic task context
\(C_e=(q_e,\dot q_e,u_e,r_e)\) from environment-specific state and action
descriptions. The LLM is prompted with this context and must emit executable
symbolic laws over flat vectors `state[i]`, `action[j]`, and `delta[k]`.

ANSWER composes the accepted laws as a weighted product of local symbolic
effects and then wraps the symbolic candidate with a neural ODE residual:

```text
symbolic candidate:   p_phi(delta | s, a, C_e) proportional to product_l p_l(delta | s, a) ^ alpha_l
neural correction:    ds/dt = f_theta(s, a, symbolic_candidate)
final prediction:     s_next = ODESolve(s, a, symbolic_candidate)
```

The graph layer is explicit but lightweight. Semantic concepts such as position,
velocity, angle, actuator, contact, and target-error act as leaders; executable
laws are followers; and state dimensions are sinks. A law can be attached to
multiple concepts, giving a small DAG:

```text
Concept -> Law -> StateDimension
```

This graph controls which laws are allowed to influence each dimension. A soft
gate keeps the neural ODE safe: symbolic effects are useful only when the
supervised dynamics loss supports them, instead of being forced into the model.

## Environments

Run the MuJoCo v5 tasks from small to large:

```text
Swimmer-v5
InvertedDoublePendulum-v5
Reacher-v5
Hopper-v5
Walker2d-v5
HalfCheetah-v5
```

`InvertedPendulum-v5` is kept as the smoke/debug task but excluded from the main
paper table because it is too easy and makes the wide table longer without
adding much evidence. `Ant-v5` and `Pusher-v5` are also excluded from the main
sweep; they need separate semantic task specs and are better treated as future
stress tests.

## Models

The final table uses short row labels to save space and starts directly from
the variant column. The experimental-design text should explain the full method
behind each label. The formatter places baselines first, ablations second, and
the proposed method last:

```text
onelife           OneLife     OneLife-MuJoCo binned LLM law mixture
pets_ensemble     PETS        PETS-style bootstrap neural ensemble
dreamer_v3        DreamerV3   External latent world-model baseline
neural            ODE-only    ANSWER without symbolic laws
program_only      LLM-only    ANSWER symbolic laws without neural ODE
symbolic_neural   Lib+ODE     Symbolic library laws plus neural ODE, no LLM laws
neural_mlp        MLP-only    Pure MLP dynamics appendix baseline
answer_mlp        ANS-MLP     ANSWER symbolic graph/gate with the MLP backbone
answer            ANSWER      Proposed neuro-symbolic ODE world model
```

`dreamer_v3` is external in this repo. Import compatible JSON results with
`model: "dreamer_v3"` and the same `score` / `reward` schema before formatting.

## Evaluation

Each environment reports:

- **R2@1**: per-dimension delta R2 for one-step prediction,
  `s_next - s`.
- **R2@10**: ten-step open-loop delta R2,
  `s_{t+10} - s_t`.
- **Reward**: real environment return after planning with the learned model.

The paper formatter also appends **Avg. Rank**. For each displayed environment,
models are ranked separately by `R2@1`, `R2@10`, and `Reward`, with higher
values better. Avg. Rank is the mean over these per-metric ranks, so lower is
better. This avoids averaging raw Reward values across MuJoCo tasks with
different scales. Table headers show this directly with `↑` for higher-is-better
metrics and `↓` for Avg. Rank.

The final config uses **PEC-CEM-MPC** as the common planner. If an output
directory also contains plain CEM-MPC results, the formatter can show both as
`CEM-MPC / PEC-CEM-MPC`.

The OneLife comparison is fair as an adapted in-repo baseline: it uses the same
MuJoCo train/test transitions, seed, LLM model, sample count, discretizer fitted
on the training set, R2 metrics, and planner evaluation protocol. The caveat is
conceptual rather than procedural: OneLife-MuJoCo remains a binned symbolic law
mixture, while ANSWER is continuous and neural-residual based. That should be
stated clearly in the paper.

## Setup

Use `uv` with Python 3.12:

```powershell
uv sync --python 3.12
```

For Windows RTX 3050/3060 machines, this CUDA wheel has worked:

```powershell
uv pip uninstall torch torchvision torchaudio
uv pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

For rented RTX 50-series machines, use the current official PyTorch CUDA wheel,
for example:

```powershell
uv pip uninstall torch torchvision torchaudio
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

After installing the CUDA wheel, run experiments with `--no-sync`; otherwise
`uv` may restore a CPU-only torch from the lockfile.

Check GPU visibility:

```powershell
uv run --no-sync python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Put API keys in `.env` on the machine that runs LLM-based models.

## Run

Smoke test:

```powershell
uv run --no-sync --env-file .env python main.py --config-name smoke
```

Full in-repo sweep, three seeds:

```powershell
uv run --no-sync --env-file .env python main.py -m problem=Swimmer-v5,InvertedDoublePendulum-v5,Reacher-v5,Hopper-v5,Walker2d-v5,HalfCheetah-v5 model=answer,onelife,pets_ensemble,neural,program_only,symbolic_neural,neural_mlp,answer_mlp seed=0,1,2 device=cuda skip_existing=true output_dir=outputs_final_answer_3seed
```

Expected in-repo JSON count:

```text
6 environments * 8 runnable models * 3 seeds = 144 JSON files
```

If the original seven-model sweep already exists, add only the fair MLP-backbone
ANSWER variant:

```powershell
uv run --no-sync --env-file .env python main.py -m problem=Swimmer-v5,InvertedDoublePendulum-v5,Reacher-v5,Hopper-v5,Walker2d-v5,HalfCheetah-v5 model=answer_mlp seed=0,1,2 device=cuda skip_existing=true output_dir=outputs_final_answer_3seed
```

Format the paper table:

```powershell
$files = Get-ChildItem outputs_final_answer_3seed -Filter *.json | Select-Object -ExpandProperty FullName
uv run --no-sync --env-file .env python scripts/format_mujoco_paper_tables.py $files --no-std
```

Export LaTeX:

```powershell
$files = Get-ChildItem outputs_final_answer_3seed -Filter *.json | Select-Object -ExpandProperty FullName
uv run --no-sync --env-file .env python scripts/format_mujoco_paper_tables.py $files --no-std --format latex | Tee-Object mujoco_main_table.tex
```

Aggregate LLM calls:

```powershell
uv run --no-sync --env-file .env python scripts/aggregate_mujoco_results.py outputs_final_answer_3seed/*.json --metric llm_calls --show-std
```

Useful diagnostics are saved under `program_residual`:

```text
one_step_delta_nll
mean_symbolic_gate
gate_active_fraction_0.01
mean_graph_budget
symbolic_gain_delta_r2_uniform
active_law_count_unique
```

These diagnostics show whether the symbolic component is active and whether it
improves the symbolic-conditioned candidate before the final gate.
