# SimFutures-LP: Future-Simulated Law Posterior for Continuous Control

SimFutures-LP is a model-based continuous-control framework for MuJoCo tasks
with hidden or partially observed dynamics factors. The method keeps the useful
infrastructure already present in this branch - per-environment LLM prompts,
safe law-DSL templates, and the fair PETS/CaDM benchmark pipeline - while using
a new future-simulated law-posterior world model.

The core claim is:

\[
\text{LLM laws}
\rightarrow
\text{latent law posterior}
\rightarrow
\text{wake-calibrated utility}
\rightarrow
\text{posterior-guided MPC}.
\]

The model is not designed to make \(R^2\) look good in isolation. It explicitly
learns which law-conditioned rollouts convert predictive reliability into real
planning return.

## Problem

For a MuJoCo state \(s_t=(q_t,\dot q_t)\) and action \(a_t\),

\[
s_{t+1}\sim P^\star(\cdot\mid s_t,a_t,u_t),
\]

where \(u_t\) denotes unobserved factors such as contact mode, friction,
actuator delay, external drift, or gait phase. Offline data are

\[
\mathcal D=\{(s_t,a_t,r_t,s_{t+1})\}_{t=1}^{N}.
\]

The control objective is

\[
A^\star=\arg\max_A
\mathbb E_{P^\star}\left[\sum_{t=0}^{H-1}\gamma^t r_t\right].
\]

The central failure mode is

\[
R^2_\theta(A)\uparrow
\centernot\Rightarrow
J^\star(A)\uparrow.
\]

SimFutures-LP targets this prediction-to-control gap directly.

## Executable Laws

An LLM law is not free-form text. It is a safe executable template:

\[
\ell_j=(V_j,\varphi_j,\sigma_j,\kappa_j).
\]

Here \(V_j\subseteq\{s,a,r,s'\}\), \(\varphi_j\) is compiled by the local DSL,
\(\sigma_j\) is a law type such as actuation, damping, inertia shift, delay, or
impulse, and \(\kappa_j\in[0,1]\) is prior confidence.

Each law produces a measurable channel:

\[
m_{j,t}=\varphi_j(s_t,a_t,s_{t+1}).
\]

In code, the LLM emits only JSON fields such as masks, `law_type`, gain,
confidence, and reward relevance. `src/onelife/duc_wm/law_dsl.py` compiles those
fields into bounded tensor effects.

## Model

The current implementation separates two questions that were coupled in earlier
variants:

\[
z_t^{dyn}: \text{which laws help predict state?}
\]

\[
z_t^{ctrl}: \text{which laws help choose high-return actions?}
\]

Both are inferred from the same law portfolio but have separate prior and
posterior encoders:

\[
p_\psi^{dyn}(z_t^{dyn}\mid h_t,s_t,a_t),
\quad
q_\phi^{dyn}(z_t^{dyn}\mid h_t,s_t,a_t,\Delta s_t,y_t^{dyn}),
\]

\[
p_\psi^{ctrl}(z_t^{ctrl}\mid h_t,s_t,a_t),
\quad
q_\phi^{ctrl}(z_t^{ctrl}\mid h_t,s_t,a_t,\Delta s_t,y_t^{ctrl}).
\]

The symbolic law effects are not added directly to the next-state delta. They
are used as conditioning features for a phase-consistent local dynamics model.
First, the history-state-action tuple is embedded into a compact latent phase:

\[
\chi_t=f_\omega(h_t,s_t,a_t).
\]

This phase is trained to be predictive over time:

\[
\hat\chi_{t+1}=g_\omega(\chi_t,s_t,a_t),
\quad
\chi^+_{t+1}=f_\omega(h^+_t,s_{t+1},a_t),
\]

\[
\mathcal L_\chi=\|\hat\chi_{t+1}-\operatorname{sg}(\chi^+_{t+1})\|^2.
\]

The law portfolio then modulates dynamics through \(\chi_t\), rather than by
adding a brittle hand-shaped residual directly into the state:

\[
m_{t,j}=m_j(s_t,a_t,h_t),
\quad
\delta_t^L=\sum_j \alpha_{t,j}^{dyn}\gamma_jm_{t,j}.
\]

\[
\pi_t=\operatorname{softmax}(f_c(\chi_t)).
\]

\[
(\Delta_{t,c},\log v_{t,c})
=
F_c(h_t,s_t,a_t;\operatorname{FiLM}(\alpha_t^{dyn},\delta_t^L,\chi_t)).
\]

\[
\widehat{\Delta s}_t
=
\sum_c \pi_{t,c}\Delta_{t,c}.
\]

\[
\hat s_{t+1}=s_t+\widehat{\Delta s}_t.
\]

This is the main structural change: laws modulate the dynamics manifold, rather
than injecting a brittle hand-shaped residual directly into the predicted state.
The same \(\chi_t\) is also passed to reward and reliability heads with
stop-gradient, so utility can use the learned phase without corrupting state
prediction.

The model also trains the deployable prior path explicitly:

\[
\mathcal L_{prior\_path}
=
-\log p_\theta(s_{t+1}\mid h_t,s_t,a_t),
\]

where the forward pass does not use \(s_{t+1}\).

## Wake-Calibrated Utility

The utility signal is trained from offline wake data, not from test rollouts.
For each transition:

\[
e_t^s=\|w\odot(\hat s_{t+1}-s_{t+1})\|^2,
\quad
e_t^r=|\hat r_t-r_t|,
\quad
e_t^l=\|\hat y_t-y_t^{ctrl}\|^2.
\]

\[
u_t
=
z(r_t)
-\lambda_s e_t^s
-\lambda_r e_t^r
-\lambda_l e_t^l.
\]

The reliability critic learns:

\[
b_t=f_B(\operatorname{sg}[s_t,a_t,\alpha_t^{ctrl},
\widehat{\Delta s}_t,\delta_t^L,\chi_t]),
\]

\[
\mathcal L_{utility}=(b_t-u_t)^2.
\]

The stop-gradient is intentional: utility learning can guide the planner, but it
does not backpropagate through and damage the predictive dynamics model.

After each epoch, the global control posterior is updated from utility evidence:

\[
e_j=\mathbb E_t[\alpha_{t,j}^{ctrl}(u_t-\bar u)].
\]

\[
\lambda_j
\leftarrow
(1-\eta)\lambda_j
+\eta\left[
\operatorname{logit}(c_j)+\frac{\bar e_j}{T}
\right].
\]

## Posterior-Guided MPC

At planning time, CEM/MPC uses the same predicted dynamics \(\hat s_{t+1}\) that
is evaluated by rollout \(R^2\). There is no separate planning-only state delta.
Action sequences are scored with predicted reward and the learned reliability
critic:

\[
S_z(A)
=
\hat J_\theta(A,z)
+\lambda_\rho\rho_\xi(z,A,h)
-\lambda_u\hat\sigma_\theta(A,z)
\]

In code this is:

\[
G(A)=\sum_t[\hat r_t+\lambda_\rho b_t-\lambda_u\hat\sigma_t].
\]

The planner selects

\[
A^\star=\arg\max_A\mathbb E_{z\sim\nu_n}[S_z(A)].
\]

In code, this is implemented as `planning.model_bonus_weight`: baselines do not
emit `planning_bonus`, so the bonus is zero for PETS/CaDM, while SimFutures uses
its reliability critic as part of the proposed method.

## Baselines

The fair comparison set remains:

| Method | Role |
|---|---|
| `duc_wm` | SimFutures-LP, the proposed method |
| `cadm_supervised` | strongest CaDM-style context-supervised baseline |
| `pets_context` | PETS-style ensemble with oracle context diagnostic |
| `cadm_context` | CaDM with oracle context diagnostic |
| `cadm` | unsupervised CaDM-style latent context |
| `pets` | PETS-style probabilistic ensemble |
| `mlp` | black-box Gaussian dynamics model |

`cadm_supervised` is the closest direct baseline because it can exploit context
supervision during training, then infer context from history during planning.
SimFutures should be evaluated on the same data, seeds, planner budget, reward
model, and MuJoCo variants.

## Code Map

```text
main.py
  Hydra entrypoint and fair method dispatch.

src/onelife/duc_wm/simfutures.py
  SimFutures-LP world model, law posterior update, training loop.

src/onelife/duc_wm/llm_prior.py
  Per-environment prompt and strict JSON law-template parser.

src/onelife/duc_wm/templates.py
  Fallback MuJoCo law templates and metadata.

src/onelife/duc_wm/law_dsl.py
  Safe compiler from law_type/masks/gains to executable tensor channels.

src/onelife/duc_wm/baselines.py
  MLP, PETS-style, and CaDM-style baselines.

src/onelife/duc_wm/planner.py
  Shared CEM-MPC planner with optional SimFutures reliability bonus.

src/onelife/duc_wm/planning_eval.py
  Executes MPC in MuJoCo extension environments.

scripts/run_balanced_gpu_jobs.py
  Parallel per-method launcher.

scripts/run_workshop_suite.py
  Multi-env, multi-seed benchmark launcher.

scripts/aggregate_avg_rank.py
  Composite AvgRank over R2@1, R2@10, and planner return.
```

## Smoke Test

```bash
uv run python main.py --config-name smoke
```

## One-Environment Fair Run

```bash
uv run python scripts/run_balanced_gpu_jobs.py \
  --config-name swimmer_full_adaptive \
  --models duc_wm,cadm_supervised,pets_context,cadm_context \
  --max-parallel 2 \
  seed=0 \
  device=cuda \
  runtime.precision=bf16 \
  runtime.preload_to_device=true \
  mujoco_extension.parallel_workers=12 \
  output_dir=outputs/simfutures_swimmer_seed0
```

## Main 5-Environment Run

```bash
uv run python scripts/run_workshop_suite.py \
  --envs swimmer,hopper,walker2d,halfcheetah,inverted_double_pendulum \
  --seeds 0,1,2 \
  --models duc_wm,cadm_supervised,pets_context,cadm_context \
  --max-parallel 3 \
  --output-dir outputs/simfutures_workshop_5env_3seed \
  --log-root outputs/balanced_logs/simfutures_workshop_5env_3seed \
  device=cuda \
  runtime.precision=bf16 \
  runtime.preload_to_device=true \
  planning.uncertainty_weight=0.0 \
  mujoco_extension.parallel_workers=12
```

## Aggregate Results

```bash
uv run python scripts/aggregate_avg_rank.py \
  "outputs/simfutures_workshop_5env_3seed/*.json" \
  --metrics score.r2_at_1,score.r2_at_10,score.planner_return_mean \
  --csv-out outputs/simfutures_workshop_5env_3seed/avg_rank_composite.csv \
  | tee outputs/simfutures_workshop_5env_3seed/avg_rank_composite.txt
```

The compact result table should report:

\[
R^2@1,\qquad R^2@10,\qquad \text{Reward},\qquad \text{AvgRank}.
\]

## LLM Prior Check

The output JSON stores both the prompt and status:

```json
"llm_prior_status": {
  "source": "llm:openai/gpt-4.1-mini",
  "error": null
}
```

Fallback is explicit:

```json
"source": "fallback_after_llm_error"
```
