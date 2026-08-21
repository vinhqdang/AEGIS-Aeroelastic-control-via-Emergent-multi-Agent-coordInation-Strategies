# AEGIS

**A**eroelastic control via **E**mergent multi-**A**gent coord**I**nation **S**trategies

Distributed flutter suppression where every trailing-edge control surface is an
independent reinforcement-learning agent with local sensing and no central
coordinator.

The research question is not "which MARL algorithm?" It is: *what does the
physics of a coupled continuum instability tell us about credit assignment and
communication that a generic MARL algorithm has to learn the hard way?*

See [`docs/ALGORITHM.md`](docs/ALGORITHM.md) for the algorithm design (MoCCA) and
the propositions it rests on.

---

## Status

| Component | State |
|---|---|
| Aeroelastic plant (modal, unsteady strip theory) | done, validated |
| Actuator bank with lag, rate/travel limits, jam failures | done |
| Distributed sensing model | done |
| Gust models (1-cosine, Dryden) | done, calibrated |
| Centralised LQR + local-feedback baselines | done |
| Robustness analysis: where the fixed gain fails | done |
| PettingZoo multi-agent environment | done |
| Animation / explainer renderer | done |
| Closed-form difference reward (Proposition 1) | derived, numerically verified |
| MoCCA policy learning | next |

## Plant validation

The plant is pinned against the published Goland cantilever wing benchmark, not
against numbers this code once produced:

| Quantity | AEGIS | Published |
|---|---|---|
| 1st bending frequency | 7.664 Hz | 7.66 Hz |
| 1st torsion frequency | 15.232 Hz | 15.24 Hz |
| Flutter speed | 137.35 m/s | ≈137 m/s |
| Flutter frequency | 69.3 rad/s | ≈70.7 rad/s |

56 plant states: 4 modal coordinates (2 bending, 2 torsion) plus 2 Wagner wake-lag
states on each of 24 aerodynamic strips.

### Model chain

1. **Structure** — Galerkin projection of a uniform cantilever wing onto exact
   clamped-free bending and fixed-free torsion modes, with all generalized
   mass/stiffness integrals evaluated by Gauss–Legendre quadrature.
2. **Aerodynamics** — strip theory with the full Theodorsen non-circulatory
   terms, and the circulatory lag realised in the time domain via the R.T. Jones
   two-exponential approximation of the Wagner function.
3. **Control surfaces** — quasi-steady thin-airfoil plain-flap derivatives from
   Glauert's series, integrated over each surface's spanwise band. Closed-form
   coefficients are cross-checked against numerical quadrature in the tests.

Known simplification: the flap's own wake shedding is neglected, which slightly
over-estimates control authority at high frequency. Documented at the point of
use, and the plan is to re-validate final policies on SHARPy (UVLM + nonlinear
beam) as the high-fidelity check.

## The core idea in one derivation

Differentiating the aeroelastic energy along trajectories of
`M q̈ + C q̇ + K q = L x_wake + B δ` gives

```
Ė = −q̇ᵀC q̇  +  q̇ᵀL x_wake  +  Σₖ (q̇ᵀbₖ) δₖ
      dissipation    wake exchange    control power — additive across agents
```

The only term depending on agent *k*'s deflection is its own control power.
So the Wolpert–Tumer difference reward has a **closed form** — no counterfactual
rollout, no learned mixing network, no centralised critic:

```
Dₖ = G(z) − G(z₋ₖ) = −(q̇ᵀbₖ) δₖ
```

`tests/test_credit_assignment.py` verifies this against brute-force
counterfactuals to 1e-10, including the absence of cross-agent terms.

This exact signal is *myopic*, though — cross-mode coupling means an agent may
need to inject energy into one mode so another can extract more from the coupled
pair. MoCCA therefore uses **exact-plus-residual** credit: physics supplies the
part it explains exactly, and a small learned head covers only the residual.

## Where centralised LQR fails

`scripts/robustness_sweep.py` quantifies the target. Gain synthesised at
U = 1.10 U_f, all numbers at sea level.

**Off-design airspeed** — the fixed gain has an airspeed margin of **1.40 U_f**.
It holds its design point comfortably (growth −5.9 /s) and is unstable by
1.50 U_f (+9.2 /s). Static divergence sits at 2.60 U_f, so this is genuine
controller failure, not unavoidable divergence.

**Lost authority** — losing any *single* surface is survivable. Losing
`aileron_mid` + `tab_outboard` together collapses the margin to **1.02 U_f** and
the closed loop is unstable at the design point (+2.7 /s).

**Finite authority** — with real travel and rate limits, the surviving jam angle
depends sharply on the control-effort weight, and **no weight survives 15°**:

| effort weight | 0° | 4° | 8° | 12° | 15° |
|---|---|---|---|---|---|
| 2e2 | −6.54 | −0.11 | div | div | div |
| 2e3 | −6.29 | −0.08 | −0.07 | div | div |
| 2e4 | −3.56 | −0.24 | −0.16 | −0.13 | div |
| 2e5 | −3.58 | div | div | div | div |

(energy growth rate, 1/s; `div` = left the linear regime)

A caveat the analysis is explicit about: **a jam angle does not move the
closed-loop spectrum.** A held deflection is an affine input — it shifts trim,
not the eigenvalues. Jam angle degrades performance through *saturation*, which
only a time-domain rollout can see. Reporting an eigenvalue margin as a function
of jam angle would be wrong, so the two studies are kept separate.

## Multi-agent environment

`aegis.envs.FlutterSuppressionEnv` is a PettingZoo `ParallelEnv`, one agent per
surface, passing `parallel_api_test`.

Observations are genuinely local: each agent gets its own accelerometer pair,
local plunge rate, twist and twist rate, its own saturation fraction, and air
data (which really is broadcast on an aircraft). No agent sees the modal state.

Two ablation axes are built in, because they *are* the experiments:

| `comm_mode` | meaning |
|---|---|
| `none` | no communication at all |
| `neighbor_local` | inboard/outboard neighbours' raw local observations |
| `modal_oracle` | true critical-mode amplitude and rate — an upper bound, not a proposed method |

| `reward_mode` | meaning |
|---|---|
| `shared` | one global scalar, the naive baseline |
| `physics` | the closed-form difference reward of Proposition 1 |
| `blended` | exact-plus-residual, what MoCCA proposes |

Episodes randomise flight condition over `speed_ratio_range` (default
1.00–1.60 U_f), initial excitation, turbulence, and optional actuator jams —
i.e. exactly the envelope the fixed-gain LQR cannot cover.

## Quick start

```bash
conda activate py313
pip install -e ".[dev]"

pytest -q                                    # 68 tests, ~15 s
python scripts/animate_flutter.py            # writes media/*.mp4
python scripts/robustness_sweep.py           # tables + media/robustness_map.png
```

`scripts/animate_flutter.py` renders four explainer animations at the same
supercritical condition (U/U_f = 1.10):

| Video | Shows |
|---|---|
| `open_loop` | flutter diverging — leaves the linear regime in 0.93 s |
| `lqr` | centralised full-state LQR, the performance ceiling |
| `local` | local rate feedback with no communication, the distributed floor |
| `lqr_jam` | LQR with the outboard tab jammed at 8°, controller not told |

Each frame pairs the deforming wing with tip plunge, per-surface deflection,
structural energy, a bending–torsion phase portrait (an orbit that opens out
*is* flutter), and the per-agent control-power bars — so credit assignment is
something you watch rather than infer from a learning curve.

## Layout

```
aegis/
  physics/
    wing.py           geometry, structure, control-surface layout, sign conventions
    modes.py          clamped-free bending and fixed-free torsion mode shapes
    thin_airfoil.py   Glauert flap derivatives (closed form + quadrature check)
    aeroelastic.py    modal assembly, Wagner wake states, state space, flutter search
  actuators.py        servo lag, rate/travel limits, jam failures
  sensors.py          per-station accelerometer pairs — deliberately partial observation
  gusts.py            1-cosine discrete gust, Dryden turbulence
  simulator.py        joint plant+actuator RK4 rollout, records per-agent control power
  control/lqr.py      centralised LQR and local-rate-feedback baselines
  analysis/robustness.py  stability margins, lost authority, saturation limits
  envs/flutter_marl.py    PettingZoo ParallelEnv, one agent per surface
  viz/                axonometric wing renderer, animation, palette
docs/ALGORITHM.md     MoCCA design, propositions, experiment plan, limitations
scripts/              runnable entry points
tests/                plant validation and Proposition 1 verification
```

## Conventions

Plunge positive **down**, twist positive **nose-up**, deflection positive
**trailing-edge down**, lift positive up, pitching moment about the elastic axis
positive nose-up. The plunge-down choice is the classical Theodorsen/Fung one,
which keeps the unsteady aerodynamic operators in textbook form.
