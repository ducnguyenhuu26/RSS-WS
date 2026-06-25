# SimFutures-LP: Future-Simulated Law Posterior for Continuous Control

SimFutures-LP is a model-based continuous-control framework for MuJoCo tasks
with hidden or partially observed dynamics factors. The method keeps the useful
part of the old DUC branch - per-environment LLM prompts, safe law-DSL
templates, and the fair PETS/CaDM benchmark pipeline - but replaces the old
mechanism-residual world model.

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

The LLM/fallback prompt creates a law hyperprior:

\[
H(c,z)=\rho(c)p_{\eta_c}(z),
\]

where \(c\) selects a prior source and \(z\in\mathbb R^K\) represents law
validity/strength.

The neural model learns:

\[
p_\psi(z_t\mid h_t,s_t,a_t),
\]

\[
q_\phi(z_t\mid h_t,s_t,a_t,s_{t+1}),
\]

\[
p_\theta(\Delta s_t,r_t\mid s_t,a_t,z_t),
\]

and a law-channel observation model

\[
p_\omega(m_t\mid z_t).
\]

The training objective is a compact joint likelihood:

\[
\mathcal L
=
\mathbb E_{q_\phi}
\left[
-\log p_\theta(\Delta s_t,r_t\mid s_t,a_t,z_t)
-\lambda_m\log p_\omega(m_t\mid z_t)
\right]
\]

\[
+
\beta D_{\mathrm{KL}}
\left(
q_\phi(z_t\mid h_t,s_t,a_t,s_{t+1})
\|
p_\psi(z_t\mid h_t,s_t,a_t)
\right)
\]

\[
+
\beta_0D_{\mathrm{KL}}
\left(
p_\psi(z_t\mid h_t,s_t,a_t)
\|
p_{\eta_c}(z_t)
\right).
\]

This keeps dynamics learning clean: no stack of conflicting mechanism-residual
losses.

## Wake-Calibrated Law Posterior

For a law-conditioned rollout, SimFutures records:

\[
(\hat\tau_i,\tau_i,z_i,A_i).
\]

Prediction error:

\[
e_i=
\frac{1}{H}\sum_{t=1}^{H}\|\hat s_t-s_t\|_2^2.
\]

Reward gap:

\[
g_i=|\hat J_i-J_i|.
\]

Law violation:

\[
v_i=\frac{1}{H}\sum_{t=0}^{H-1}\|m_t\|_2^2.
\]

Wake utility:

\[
U_i(z_i,A_i)
=
J_i-\lambda_e e_i-\lambda_g g_i-\lambda_v v_i.
\]

The posterior update is trust-region regularized:

\[
\nu_{n+1}
=
\arg\max_{\nu}
\left[
\mathbb E_{z\sim\nu}\widehat U_n(z)
-\tau D_{\mathrm{KL}}(\nu\|\nu_n)
-\lambda D_{\mathrm{KL}}(\nu\|H)
\right].
\]

Its closed-form update is:

\[
\nu_{n+1}(z)
\propto
\nu_n(z)^{\frac{\tau}{\tau+\lambda}}
H(z)^{\frac{\lambda}{\tau+\lambda}}
\exp\left(\frac{\widehat U_n(z)}{\tau+\lambda}\right).
\]

The implementation uses an online approximation of this update after each
training epoch.

## Posterior-Guided MPC

At planning time, CEM/MPC evaluates action sequences with both predicted reward
and the learned reliability critic:

\[
S_z(A)
=
\hat J_\theta(A,z)
+\lambda_\rho\rho_\xi(z,A,h)
-\lambda_u\hat\sigma_\theta(A,z)
-\lambda_v\hat v(A,z).
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
