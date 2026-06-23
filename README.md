# DUC-WM: Disentangled Universal Causal World Model

This repository now implements **DUC-WM**, a new model-based control framework
for hidden-mechanism continuous-control environments.

DUC-WM is designed for settings where the next state is not caused only by the
current state and action. Deployment can include wind, friction shift, actuator
delay, sticky actions, sticky transitions, mass/damping changes, or impulse
forces. The goal is not to claim that these disturbances are new. The goal is to
learn a world model that explains which mechanism is active and how strongly it
affects control.

## Problem

We consider a family of continuous-control environments:

$$
e\sim p(e)
$$

with state and action:

$$
x_t\in\mathbb R^d,\qquad a_t\in\mathbb R^m
$$

The true transition is:

$$
x_{t+1}=F^\star_e(x_t,a_t)+\epsilon_t
$$

The environment instance \(e\) is not directly observed. The agent only sees a
short history:

$$
h_t=(x_{t-L},a_{t-L},\dots,x_t)
$$

The control objective is:

$$
\max_\pi\mathbb E_e[J_e(\pi)]
$$

where:

$$
J_e(\pi)=
\mathbb E\left[
\sum_{t=0}^{T}\gamma^t r_e(x_t,a_t)
\right]
$$

DUC-WM targets hidden-mechanism dynamics:

$$
x_{t+1}=F^\star(x_t,a_t,c_e)+\epsilon_t
$$

where \(c_e\) is an unobserved context describing the strength of external or
latent mechanisms.

## Core Assumption

DUC-WM models the state delta:

$$
\Delta x_t=x_{t+1}-x_t
$$

as a sum of universal causal mechanisms:

$$
\Delta x_t=
\sum_{j=1}^{K}\alpha_{j,t}M_j(x_t,a_t)+\epsilon_t
$$

Here:

| Symbol | Meaning |
|---|---|
| \(M_j\) | universal causal mechanism |
| \(\alpha_{j,t}\) | instance-specific mechanism strength |
| \(\alpha_{j,t}M_j\) | contribution of mechanism \(j\) |
| \(\epsilon_t\) | unmodeled noise |

The mechanism is invariant:

$$
M_j^{(e_1)}\approx M_j^{(e_2)}
$$

but the strength adapts:

$$
\alpha_{j,t}=\alpha_j(h_t)
$$

Example mechanisms for Ant:

| Mechanism | Interpretation |
|---|---|
| actuation | action produces joint/body velocity change |
| wind | external drift affects body velocity |
| friction | contact/friction changes slip and velocity response |
| mass | inertia changes acceleration induced by control |
| damping | passive dissipation changes rollout stability |
| delay | current state responds to stale actions |
| sticky | transition is partially stuck near previous state |
| impulse | rare unmodeled force causes sudden velocity change |
| gravity | passive acceleration and balance shift |

## Why Additive Bounded Mechanisms

DUC-WM uses:

$$
\hat{\Delta x}_t=
\sum_j \alpha_{j,t}M_j(x_t,a_t)
$$

and not a product of mechanism outputs. Products are difficult for state deltas
because deltas can be negative, long rollouts can explode, and attribution
becomes unclear. Additive mechanisms have stable gradients:

$$
\frac{\partial\hat{\Delta x}}{\partial M_j}=\alpha_j
$$

and:

$$
\frac{\partial\hat{\Delta x}}{\partial \alpha_j}=M_j
$$

This makes it easier to learn both the shape of a mechanism and its current
strength.

## Offline LLM Prior

The LLM is not called in the training loop. It is used offline to construct a
prior over plausible mechanisms.

For each mechanism, the LLM prior is represented as:

$$
T_j=(P_j^x,P_j^a,O_j,\Omega_j,s_j,\rho_j)
$$

| Component | Meaning |
|---|---|
| \(P_j^x\) | state dimensions read by mechanism \(j\) |
| \(P_j^a\) | action dimensions read by mechanism \(j\) |
| \(O_j\) | state dimensions affected by mechanism \(j\) |
| \(\Omega_j\) | plausible range for \(\alpha_j\) |
| \(s_j\) | maximum bounded scale |
| \(\rho_j\) | prior confidence |
| reward relevance | short explanation of why the mechanism matters for planning |

The prior is:

$$
p_{\mathrm{LLM}}(M,c)
$$

where:

$$
c=(\alpha_1,\dots,\alpha_K)
$$

The implementation includes an environment-specific prompt builder in
`onelife.duc_wm.llm_prior`. Prompts differ across Ant, Walker2d, Hopper,
Reacher, Pusher, Swimmer, HalfCheetah, InvertedPendulum, and
InvertedDoublePendulum profiles. Each prompt tells the LLM the task objective,
reward-critical errors, state/action index contract, allowed mechanism
families, and strict JSON schema. The prompt is stored in each output JSON under
`llm_prior_prompt` for traceability.

There are three prior modes:

| Mode | Config | Use |
|---|---|---|
| deterministic fallback | `duc.llm_prior.enabled=false` and `json_path=null` | reproducible baseline |
| direct LLM generation | `duc.llm_prior.enabled=true` | generate a candidate prior |
| saved LLM prior | `duc.llm_prior.json_path=path/to/prior.json` | paper/benchmark run |

For fair comparison, generate a prior once, save the JSON, then run all seeds
with the same `json_path`.

## Model

Each mechanism is a small MLP:

$$
M_{\theta,j}:
(x_{P_j^x},a_{P_j^a})\mapsto \Delta x_{O_j}
$$

The local output is expanded to the full state dimension:

$$
\tilde M_{\theta,j}(x_t,a_t)
=
O_j\odot M_{\theta,j}(x_{P_j^x},a_{P_j^a})
$$

The bounded strength is:

$$
\alpha_{j,t}=s_j\tanh(u_{j,t})
$$

so:

$$
|\alpha_{j,t}|\le s_j
$$

The predicted next state is:

$$
\hat x_{t+1}
=
x_t+
\sum_{j=1}^{K}
\alpha_{j,t}\tilde M_{\theta,j}(x_t,a_t)
$$

The probabilistic head is:

$$
p_\theta(x_{t+1}\mid x_t,a_t,c_t)
=
\mathcal N(\mu_\theta,\Sigma_\theta)
$$

with:

$$
\mu_\theta=\hat x_{t+1}
$$

and diagonal variance:

$$
\Sigma_\theta=\mathrm{diag}(\mathrm{softplus}(S_\theta(x_t,a_t,c_t)))
$$

## Context Inference

During real deployment, the context \(c_t\) is not known. DUC-WM infers it from
history:

$$
q_\phi(c_t\mid h_t)
=
\mathcal N(
\mu_\phi(h_t),
\mathrm{diag}(\sigma_\phi^2(h_t))
)
$$

Prediction marginalizes over context:

$$
p(x_{t+1}\mid x_t,a_t,h_t)
=
\int
p_\theta(x_{t+1}\mid x_t,a_t,c)
q_\phi(c\mid h_t)\,dc
$$

The implementation uses a small Monte Carlo approximation when needed.

## Virtual Causal Space

DUC-WM samples virtual contexts:

$$
\mathcal V=\{c_1,\dots,c_N\}
$$

from the LLM/context prior:

$$
c_i\sim p_{\mathrm{LLM}}(c)
$$

Each context specifies the strength of mechanisms such as wind, friction,
damping, delay, sticky transition, impulse, and gravity shift. The MuJoCo
extension collector saves:

$$
\mathcal D_{\mathcal V}
=
\{x_t,a_t,x_{t+1},c_i,r_t,d_t\}
$$

The context labels make the first implementation stable and debuggable.

## Training Objective

The runnable implementation trains the full DUC-WM objective with conservative
default weights. The goal is not to stack many losses blindly; each term maps to
one part of the framework.

$$
\mathcal L_{\mathrm{DUC}}
=
\mathcal L_{\mathrm{nll}}
+
\beta\mathcal L_{\mathrm{KL}}
+
\lambda_{\mathrm{ctx}}\mathcal L_{\mathrm{ctx}}
+
\lambda_{\mathrm{ctrl}}\mathcal L_{\mathrm{ctrl}}
$$

$$
+
\lambda_{\mathrm{roll}}\mathcal L_{\mathrm{roll}}
+
\lambda_{\mathrm{orth}}\mathcal L_{\mathrm{orth}}
+
\lambda_{\mathrm{sparse}}\mathcal L_{\mathrm{sparse}}
$$

where:

$$
\mathcal L_{\mathrm{nll}}
=
-
\mathbb E_{c\sim q_\phi(c\mid h)}
[
\log p_\theta(x'\mid x,a,c)
]
$$

The prior regularizer is:

$$
\mathcal L_{\mathrm{KL}}
=
\mathrm{KL}
(
q_\phi(c\mid h)
\|
p_{\mathrm{LLM}}(c)
)
$$

When virtual context labels exist:

$$
\mathcal L_{\mathrm{ctx}}
=
\|\mu_\phi(h)-c^\star\|_2^2
$$

The control-aware one-step term is:

$$
\mathcal L_{\mathrm{ctrl}}
=
\|W_{\mathrm{ctrl}}^{1/2}(\hat x'-x')\|_2^2
$$

The rollout term unrolls the learned model for \(H\) steps under observed
actions:

$$
\hat x_{t+k+1}
=
F_\theta(\hat x_{t+k},a_{t+k},c_{t+k})
$$

$$
\mathcal L_{\mathrm{roll}}
=
\frac{1}{H}
\sum_{k=0}^{H-1}
\|W_{\mathrm{ctrl}}^{1/2}(\hat x_{t+k+1}-x_{t+k+1})\|_2^2
$$

The orthogonal and sparse terms keep mechanisms separated and avoid using every
mechanism for every transition:

$$
\mathcal L_{\mathrm{orth}}
=
\sum_{i\ne j}
\langle W_{\mathrm{ctrl}}^{1/2}M_i,
W_{\mathrm{ctrl}}^{1/2}M_j\rangle^2
$$

$$
\mathcal L_{\mathrm{sparse}}
=
\|\alpha\|_1
$$

Default values are deliberately mild:

| Term | Default |
|---|---:|
| \(\lambda_{\mathrm{ctx}}\) | 1.0 |
| \(\lambda_{\mathrm{ctrl}}\) | 0.05 |
| \(\lambda_{\mathrm{roll}}\) | 0.1 |
| \(H\) | 5 |
| \(\lambda_{\mathrm{orth}}\) | 0.0001 |
| \(\lambda_{\mathrm{sparse}}\) | 0.00001 |

## Control-Relevant Metric

Ordinary \(R^2\) can be high while planning reward is low. DUC-WM therefore
tracks a control-weighted metric.

Ordinary one-step score:

$$
R^2
=
1-
\frac{\sum_t\|x_t-\hat x_t\|_2^2}
{\sum_t\|x_t-\bar x\|_2^2}
$$

Control-weighted score:

$$
R^2_{\mathrm{DUC}}
=
1-
\frac{\sum_t\|x_t-\hat x_t\|_{W_t}^2}
{\sum_t\|x_t-\bar x\|_{W_t}^2}
$$

where:

$$
\|v\|_{W_t}^2=v^\top W_t v
$$

In the current implementation, \(W_t\) is the stable default constructed from
mechanism output masks. If a learned reward/value model is added for a specific
benchmark, the same interface also supports gradient-derived weights:

$$
W_t=
\nabla_x V(x_t)\nabla_x V(x_t)^\top+\lambda I
$$

## Planning

The model supports MPC-style planning:

$$
a^\star_{t:t+H}
=
\arg\max_a
\mathcal J_\theta
$$

with:

$$
\mathcal J_\theta
=
\mathbb E_{c\sim q_\phi}
\left[
\sum_{\ell=0}^{H}
\gamma^\ell r(\hat x_{t+\ell},a_{t+\ell})
\right]
-\beta U
$$

Uncertainty has model and context parts:

$$
U=
U_{\mathrm{model}}+U_{\mathrm{ctx}}
$$

where:

$$
U_{\mathrm{model}}
=
\mathbb E_c[
\mathrm{tr}\Sigma_\theta(x,a,c)
]
$$

and:

$$
U_{\mathrm{ctx}}
=
\mathrm{Var}_{c\sim q_\phi}
[
F_\theta(x,a,c)
]
$$

The current planner code is a compact CEM/MPC utility. The default CLI focuses
on training and prediction metrics first.

## Theoretical Analysis

### 1. LLM Prior Bound

Let a mechanism world model be:

$$
f=(M,c)
$$

The LLM prior is:

$$
p_{\mathrm{LLM}}(f)
$$

Training produces posterior:

$$
q(f)
$$

For bounded losses, a PAC-Bayes-style statement gives, with probability at
least \(1-\delta\):

$$
\mathcal L(q)
\le
\widehat{\mathcal L}(q)
+
\mathcal O
\left(
\sqrt{
\frac{
\mathrm{KL}(q\|p_{\mathrm{LLM}})
+
\log(1/\delta)}
{n}}
\right)
$$

Interpretation: if the LLM prior places mass near useful mechanisms, the
posterior has smaller KL and the bound is tighter. If the prior is wrong, this
advantage disappears and must be shown by ablation.

### 2. Context Reduces Bayes Risk

Assume:

$$
x'=F^\star(x,a,c)+\epsilon
$$

A context-free predictor uses:

$$
f^\star_{\mathrm{noctx}}(x,a)
=
\mathbb E[x'\mid x,a]
$$

Its Bayes risk is:

$$
R^\star_{\mathrm{noctx}}
=
\mathbb E[
\|x'-\mathbb E[x'\mid x,a]\|^2
]
$$

A context-aware predictor uses:

$$
f^\star_{\mathrm{ctx}}(x,a,c)
=
\mathbb E[x'\mid x,a,c]
$$

By total variance:

$$
\mathrm{Var}(x'\mid x,a)
=
\mathbb E[
\mathrm{Var}(x'\mid x,a,c)
]
$$

$$
+
\mathrm{Var}
(
\mathbb E[x'\mid x,a,c]
\mid x,a
)
$$

Therefore:

$$
R^\star_{\mathrm{ctx}}
\le
R^\star_{\mathrm{noctx}}
$$

The inequality is strict when context changes the conditional next-state mean.
DUC-WM approaches the context-aware predictor when \(q_\phi(c\mid h)\) can infer
context from history.

### 3. Control-Weighted Error Bounds Return Gap

Assume the value function is Lipschitz in the control norm:

$$
|V(x)-V(y)|
\le
L_V\|x-y\|_{W}
$$

Then the model-return gap is bounded by:

$$
|J_{\mathrm{real}}(\pi)-J_{\mathrm{model}}(\pi)|
\le
C
\sum_t
\mathbb E[
\|x_t-\hat x_t\|_{W_t}
]
+
C'U
$$

Thus ordinary \(R^2\) is not enough. The relevant quantity is the
control-weighted error that appears in \(R^2_{\mathrm{DUC}}\).

### 4. Conditional Comparison With Context-Free Models

DUC-WM does not claim to always beat every world model. A valid comparison
statement is conditional.

Let:

$$
B_D=C E_{W,D}+C'U_D
$$

and:

$$
B_N=C E_{W,N}+C'U_N
$$

be return-gap bounds for DUC-WM and a context-free model. If:

$$
\hat J_D(\pi_D)-\hat J_N(\pi_N)>B_D+B_N
$$

then:

$$
J(\pi_D)>J(\pi_N)
$$

Proof:

$$
J(\pi_D)\ge \hat J_D(\pi_D)-B_D
$$

and:

$$
J(\pi_N)\le \hat J_N(\pi_N)+B_N
$$

The result follows by substitution. DUC-WM is expected to have smaller \(B_D\)
only when the environment has latent but inferable mechanisms.

## Runtime Envelope

The implementation is intentionally H100-friendly:

| Component | Default |
|---|---|
| mechanisms | 9 |
| MLP layers | 2 |
| hidden size | 256 |
| context encoder | small MLP over history |
| context samples | posterior mean or one sampled context in training |
| LLM use | offline only |
| planner | CEM/MPC module over the learned world model |

The LLM is not in the gradient loop.

## Run

Smoke run:

```bash
uv run --no-sync python main.py --config-name smoke
```

Default DUC-WM run:

```bash
uv run --no-sync python main.py problem=Hopper-v5 device=cuda
```

Generate a fresh LLM prior during a run:

```bash
uv run --no-sync python main.py \
  problem=Hopper-v5 \
  device=cuda \
  duc.llm_prior.enabled=true
```

Run from a saved LLM prior JSON:

```bash
uv run --no-sync python main.py \
  problem=Hopper-v5 \
  device=cuda \
  duc.llm_prior.json_path=priors/hopper_duc_prior.json
```

Example sweep:

```bash
uv run --no-sync python main.py -m \
  problem=Swimmer-v5,Reacher-v5,Hopper-v5,Walker2d-v5,Ant-v5 \
  seed=0 \
  device=cuda \
  mujoco_extension.variant=all \
  output_dir=outputs/duc_wm
```

Aggregate one metric:

```bash
uv run --no-sync python scripts/aggregate_mujoco_results.py \
  outputs/duc_wm/*.json \
  --metric score.duc_r2_at_1
```

## Current Claim

DUC-WM is for hidden-mechanism continuous control. It uses an offline LLM prior
to define plausible causal mechanisms, trains modular neural mechanisms and a
context posterior, and evaluates prediction with a control-aware metric.

The valid claim is not "DUC-WM always beats NSWM." The valid claim is:

If transition dynamics depend on latent but inferable mechanisms, a
context-aware mechanism world model can have lower Bayes transition risk than a
context-free predictor. If its control-weighted rollout error and uncertainty
are lower, its model-planning return gap is also lower.
