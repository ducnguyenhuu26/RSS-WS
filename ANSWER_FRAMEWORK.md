# ANSWER Framework

This repository now uses one proposed model name: `answer`.

`answer` combines environment-conditioned symbolic law synthesis with a neural
ODE residual. The final comparison protocol is intentionally small:

Main table:

```text
answer
onelife
pets_ensemble
dreamer_v3
```

Ablation table:

```text
neural            # no symbolic component, same ODE backbone as ANSWER
program_only      # symbolic only, no neural residual
symbolic_neural   # symbolic library + neural ODE, no LLM-generated laws
```

`dreamer_v3` is an external baseline. Import its JSON results with
`model: "dreamer_v3"` and the same `score` / `reward` schema before formatting
tables.

## Core Objects

For each MuJoCo environment \(e\), the code builds a semantic task context

\[
C_e = (\mathcal{Q}_e, \mathcal{V}_e, \mathcal{U}_e, \mathcal{G}_e),
\]

where \(\mathcal{Q}_e\) are qpos/angle-like coordinates, \(\mathcal{V}_e\) are
qvel coordinates, \(\mathcal{U}_e\) are ctrl torque/force inputs, and
\(\mathcal{G}_e\) contains target, geometry, constraint, reward, and termination
semantics. This is implemented in:

```text
src/onelife/program_residual/task_specs.py
```

The LLM prompt is rendered by:

```text
src/onelife/program_residual/llm_synthesizer.py
```

The prompt no longer asks the LLM to infer anonymous arrays. It gives explicit
labels such as `state[0]=cart_position`, `state[2]=cart_velocity`,
`action[0]=cart_force`, and grouped MuJoCo concepts such as qpos, qvel,
qfrc_constraint, and ctrl torque/force inputs.

## Symbolic Law Distribution

The LLM generates follower laws attached to semantic leader concepts. A law has
the form

\[
\ell = (I_\ell, J_\ell, T_\ell, f_\ell, w_\ell, \sigma_\ell),
\]

where \(I_\ell\) are state parent coordinates, \(J_\ell\) are action parent
coordinates, \(T_\ell\) are predicted target coordinates, \(f_\ell\) is the law
code, and \(w_\ell,\sigma_\ell\) describe law reliability.

For target coordinate \(i\), a law predicts a local probabilistic effect on the
state change

\[
m_{\ell,i}(s,a) = \psi_\ell(s^{I_\ell}, a^{J_\ell}; \alpha_\ell).
\]

ANSWER does not trust this contribution by construction. Each law receives a
nonnegative learned weight

\[
\rho_\ell = \mathrm{softplus}(\beta_\ell),
\]

and contributes a weighted likelihood

\[
p_\ell(\Delta s_i\mid s,a)
=\mathcal{N}(\Delta s_i;m_{\ell,i}(s,a),\sigma_{\ell,i}^2).
\]

Active laws are composed as a weighted product, following the OneLife-style
law-mixture view:

\[
p_i(\Delta s_i\mid s,a,\mathcal{L}_i)
\propto
\prod_{\ell\in\mathcal{L}_i}
p_\ell(\Delta s_i\mid s,a)^{\rho_\ell}.
\]

Equivalently,

\[
\log p_i
=\mathrm{const}+
\sum_{\ell\in\mathcal{L}_i}
\rho_\ell\log p_\ell(\Delta s_i\mid s,a).
\]

For ODE rollout, the normalized Gaussian product yields a symbolic delta mean
\(\mu_i^{sym}\). With a neutral zero-delta base prior, the implemented mean is:

\[
\mu_i^{sym}
=
\frac{\sum_{\ell\in\mathcal{L}_i}\rho_\ell\pi_{\ell,i}m_{\ell,i}}
     {\pi_0+\sum_{\ell\in\mathcal{L}_i}\rho_\ell\pi_{\ell,i}},
\qquad
\pi_{\ell,i}=w_{\ell,i}/\sigma_{\ell,i}^2.
\]

The law weights are initialized near zero and optimized from data with an L1
penalty. This makes the LLM a generator of candidate structure, not an
authority: unsupported laws can collapse back to zero, leaving the neural ODE
to model the transition.

## Leader-Follower Concept Graph

ANSWER represents symbolic structure as a small semantic graph:

\[
G = (V_C \cup V_L, E),
\]

where \(V_C\) are concept leaders such as qpos, qvel, ctrl/torque, constraint,
target, and geometry, while \(V_L\) are executable follower laws.

Typical edges are:

\[
qpos, qvel \rightarrow \ell_{kin} \rightarrow qpos',
\]

\[
qvel, ctrl \rightarrow \ell_{act} \rightarrow qvel',
\]

\[
target, geometry \rightarrow \ell_{id} \rightarrow target'/geometry'.
\]

This is the intended meaning of leader/follower in this repo: leaders are
semantic concepts, not a single dominant law.

## Semantic Evolution

The island search generates multiple candidate law programs. Each law receives
a semantic token set \(z_\ell\) from its law name, predicted dimensions, and
MuJoCo concept labels.

Candidate fitness is:

\[
F(P) =
R^2_\Delta(P)
+ \lambda_{cov}\mathrm{cov}(P)
+ \lambda_{focus}\mathrm{focus}(P)
- \lambda_c |P|
- \lambda_{bad}\mathrm{bad}(P).
\]

Crossover selects laws using both performance and semantic novelty:

\[
\mathrm{score}(\ell \mid S)
= F(\{\ell\})
+ \eta\left(1 - \max_{k \in S}\mathrm{sim}(z_\ell,z_k)\right),
\]

where \(\mathrm{sim}\) is Jaccard similarity. Mutation removes laws that are
both low-fitness and semantically redundant.

Implementation:

```text
src/onelife/program_residual/island_search.py
```

## Neural Residual

The final `answer` model uses a neural ODE residual around the symbolic
transition. Let

\[
v_{sym}(s,a) = \mu^{sym}(s,a;C_e)/\Delta t.
\]

ANSWER integrates:

\[
\frac{dx}{d\tau}
= v_{sym}(s,a) + f_\theta(x,a,\hat{s}^{sym},m,\tau),
\qquad x(0)=s,
\]

where \(m\) is the symbolic unknown mask. The predicted next state and
transition distribution are:

\[
\hat{s}' = x(\Delta t).
\]

\[
p(\Delta s\mid s,a,C_e)
=\mathcal{N}(\hat{s}'-s,\Sigma_\eta(s,a)).
\]

The training objective is:

\[
\min_{\theta,\eta,\alpha,\beta}
-\log p(\Delta s\mid s,a,C_e)
+\lambda_\rho\sum_\ell \rho_\ell
+\lambda_r\|r_\theta\|^2.
\]

Implementation:

```text
src/onelife/program_residual/residual.py
src/onelife/program_residual/model.py
```

## Metrics

The score column is one-step delta R2:

\[
y = s' - s,
\qquad
\hat{y} = \hat{s}' - s,
\]

\[
R^2_i = 1 - \frac{\sum_t (y_{t,i}-\hat{y}_{t,i})^2}
                  {\sum_t (y_{t,i}-\bar{y}_i)^2},
\qquad
R^2_\Delta = \frac{1}{d}\sum_i R^2_i.
\]

The reward column reports real environment return after planning with the
learned model:

```text
Reward = CEM-MPC / PEC-CEM-MPC
```

CEM-MPC is the plain cross-entropy MPC planner. PEC-CEM-MPC uses the same
planner but subtracts model-risk penalties during imagined rollouts.

## Recommended Environment Order

Run from smaller/simpler observation-action spaces to larger ones:

```text
1. InvertedPendulum-v5        state=4,  action=1
2. Swimmer-v5                 state=8,  action=2
3. InvertedDoublePendulum-v5  state=9,  action=1
4. Reacher-v5                 state=10, action=2
5. Hopper-v5                  state=11, action=3
6. Walker2d-v5                state=17, action=6
7. HalfCheetah-v5             state=17, action=6
```

This gives early feedback on the smallest systems before spending time on the
larger locomotion tasks.

## Commands

Smoke test:

```powershell
uv run --no-sync --env-file .env python main.py --config-name smoke device=cuda
```

Full in-repo sweep, three seeds:

```powershell
uv run --no-sync --env-file .env python main.py -m problem=InvertedPendulum-v5,Swimmer-v5,InvertedDoublePendulum-v5,Reacher-v5,Hopper-v5,Walker2d-v5,HalfCheetah-v5 model=answer,onelife,pets_ensemble,neural,program_only,symbolic_neural seed=0,1,2 device=cuda skip_existing=true output_dir=outputs_final_answer_3seed
```

Format final tables:

```powershell
$files = Get-ChildItem outputs_final_answer_3seed -Filter *.json | Select-Object -ExpandProperty FullName
uv run --no-sync --env-file .env python scripts/format_mujoco_final_table.py $files --no-std
```

Dreamer V3 is external. Add compatible JSON files to the same output directory
with `model: "dreamer_v3"` before running the formatter if those results are
available.
