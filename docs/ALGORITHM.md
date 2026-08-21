# MoCCA — Modal Consensus & Credit Assignment

Working name for the AEGIS algorithm. Three mechanisms, each derived from a
property of the aeroelastic plant rather than imported from game-like MARL
benchmarks.

The framing question is not "which MARL framework?" It is: *what does the physics
of a coupled continuum instability tell us about credit assignment and
communication that a generic MARL algorithm has to learn the hard way?*

---

## 0. Why generic MARL is the wrong default here

Flutter is a **global modal instability**, not a spatially local one. A specific
coupled bending–torsion eigenvalue crosses into the right half plane at a
specific frequency. Each control surface acts locally in space, but its effect on
that unstable mode is set by the *modal residue at its station* — the value of
the mode shape there.

Three consequences that generic MARL handles badly:

1. **Two agents can read identical local sensors and have opposite influence on
   the unstable mode.** Local observation does not determine useful local action.
2. **A shared scalar reward is nearly uninformative.** All three surfaces act on
   the same mode; damping is a sum. Standard learned decompositions (VDN, QMIX,
   COMA) must infer the split from noisy returns.
3. **The sign of an agent's authority flips with flight condition.** Control
   reversal is real: for the Goland section the elastic axis sits aft of the
   aerodynamic centre, so flap-down produces lift up *and* nose-down twist. Which
   term dominates depends on dynamic pressure.

MoCCA attacks each of these with structure that is available in closed form.

---

## 1. Physics-exact difference rewards

### The decomposition

The modal plant (see `aegis/physics/aeroelastic.py`) is

```
M xddot + C xdot + K x = L X_wake + B delta
```

with `x` the modal amplitudes, `X_wake` the Wagner lag states, and `B` the
control influence whose `k`-th column `b_k` belongs to surface `k`.

Take the structural mechanical energy

```
E = 1/2 xdot^T M xdot + 1/2 x^T K x
```

and differentiate along trajectories:

```
Edot = xdot^T (M xddot) + xdot^T K x
     = xdot^T (-C xdot - K x + L X_wake + B delta) + xdot^T K x
     = -xdot^T C xdot  +  xdot^T L X_wake  +  sum_k (xdot^T b_k) delta_k
```

The three terms are, in order: damping dissipation (aerodynamic plus
structural), energy exchange with the shed wake, and **control power — which is
exactly additive across agents.**

Define agent `k`'s instantaneous control power

```
P_k = (xdot^T b_k) * delta_k
```

`P_k < 0` means surface `k` is extracting energy from the structure.

### Proposition 1 (closed-form difference reward)

For a global objective whose only agent-dependent term is the control power, the
Wolpert–Tumer difference reward

```
D_k = G(z) - G(z_{-k})
```

admits a closed form and equals `-P_k`. No counterfactual rollout, no learned
mixing network, no centralised critic is needed to compute it.

*Proof sketch.* Removing agent `k` sets `delta_k = 0`. Every other term in `Edot`
is independent of `delta_k`, and the control-power term is a sum over agents, so
the difference collapses to the single term `(xdot^T b_k) delta_k`. ∎

This is the point of leverage. MARL credit assignment is normally hard *because*
the decomposition must be learned. In an energy-conserving structural plant with
additive control authority, the decomposition is handed to us.

### Modal weighting

Project onto the in-vacuo modes (`V` mass-normalised, `x = V eta`). The control
contribution to mode `r` from agent `k` is `etadot_r * (v_r^T b_k) * delta_k` —
still exactly separable in *both* agent and mode. So the credit signal can be
focused on whichever mode is actually going unstable:

```
r_k^phys  =  - sum_r  w_r * etadot_r * (v_r^T b_k) * delta_k
```

with `w_r` concentrated on the flutter-critical mode. `v_r^T b_k` is the modal
residue — the quantity that decides whether agent `k` can do anything useful at
all, and whose sign flips under control reversal.

### Why exact is not enough (and this is the real contribution)

`r_k^phys` is **myopic**. It rewards instantaneous energy extraction, but the
optimal policy is not greedy: cross-mode coupling through the aerodynamic damping
`C` and through the wake states `L X_wake` means an agent may need to *inject*
energy into one mode so another agent can extract more from the coupled pair. A
purely greedy energy-extraction reward provably cannot express that.

So MoCCA uses **exact-plus-residual credit**:

```
r_k  =  r_k^phys  +  f_theta(o_k, m_consensus)
```

where `r_k^phys` carries the part of credit the physics explains exactly (low
variance, unbiased, free) and a small learned residual head `f_theta` captures
only the cross-mode coupling the energy budget cannot see. Equivalently: use the
modal energy `Phi = -sum_r w_r E_r` as a potential-based shaping term (policy
invariant, Ng et al. 1999) and let the critic learn only the residual advantage.

**Claim to test:** exact-plus-residual dominates both pure learned credit
(high variance) and pure physics credit (biased toward greedy).

---

## 2. Spectral consensus communication

### The message is a modal phasor

To coordinate, the agents do not need to exchange observations. They need to
**agree on the phase and amplitude of the handful of modes they are all
fighting.**

Each agent runs a local recurrent encoder over its own accelerometer pair
producing an estimate of the critical-mode phasors

```
m_k = { (A_r, phi_r) }  for r in the retained critical set,  |retained| = R
```

then agents run `T` rounds of averaging consensus over the communication graph
(nearest-neighbour along the span — physically what a distributed avionics bus
looks like) to reach approximate agreement.

Bandwidth: `2R` floats per agent per step, **independent of the number of
surfaces**. That is what makes the scaling story credible.

### Proposition 2 (sufficiency under modal truncation)

For the linear plant the centralised optimal control is `delta* = -K_lqr x`. If
the retained modal set spans the controllable critical subspace, then the union
of (local station measurements) and (consensus phasors of the retained modes)
determines `delta*` up to a residual bounded by the truncated modes' energy
contribution. Stated honestly: this is a sufficiency condition *with an explicit
truncation bound*, not an exact equivalence.

### Why this beats learned messages

CommNet / TarMAC / IC3Net learn arbitrary message vectors. Structuring the
message space as a modal phasor buys three things a learned vector does not:

- **Interpretability.** You can plot what the agents are saying, and it is a
  physical quantity.
- **Tiny, fixed bandwidth.** `2R` floats regardless of agent count.
- **Graceful degradation you can characterise.** Sweep consensus rounds `T`,
  packet-drop rate, and channel delay — all directly relevant to real
  distributed flight-control hardware, and all meaningless for an
  uninterpretable learned message.

---

## 3. Phase-locked action parameterisation

Instead of emitting raw deflection, each agent emits a gain and a phase offset
*relative to the consensus phasor*:

```
delta_k  =  sum_r  A_k^r * sin(theta_r + psi_k^r)   +   u_k^broadband
```

Rationale: flutter suppression is fundamentally a resonant phase problem. Baking
the resonant structure into the action space shrinks the effective policy class
enormously and makes the learned policy directly comparable to a classical
distributed modal-damping law. The additive `u_k^broadband` channel is retained
because turbulence response is *not* narrowband — without it the parameterisation
would fail under Dryden excitation.

### The emergent behaviour to look for

With this parameterisation the thing that should emerge is a **spanwise phase
gradient** — agents spontaneously adopting a phase pattern that forms a
travelling-wave actuation matched to the mode shape, without ever being told the
mode shape. That is:

- the concrete meaning of "emergent coordination" in AEGIS,
- directly visualisable (animate `psi_k` versus span as training progresses),
- and the centrepiece figure of the paper.

---

## 4. Experimental design implied by the above

### Baselines (the two questions a reviewer will ask first)

| Baseline | Answers |
|---|---|
| Centralised LQG / H-infinity with full state | "Why not just use optimal control?" |
| Single-agent PPO with all sensors and all actuators | "Why multi-agent at all?" |
| Independent PPO per surface, shared reward | "Why not the obvious MARL baseline?" |
| MAPPO / QMIX with learned credit | "Why your credit assignment?" |

### Ablations (each isolates one mechanism)

1. Shared global reward → physics-exact credit → exact-plus-residual.
2. No comm → learned message vector → spectral phasor consensus, **at equal bandwidth**.
3. Raw deflection action → phase-locked action → phase-locked plus broadband.
4. Consensus rounds `T` = 0, 1, 2, 4; packet drop 0–50%; channel delay 0–20 ms.

### Chosen framing

Decided 2026-08-21: **robustness is the headline empirical result, Proposition 1
is the theoretical contribution.** Both halves are in scope, so the experiment
budget must cover the centralised LQR baseline plus a failure / off-design sweep
*and* the learned-credit baselines (MAPPO, QMIX, COMA). The scaling and emergence
study is secondary -- worth running, not the claim the paper stands on.

### The headline claim

> Distributed learned control **matches** centralised optimal control at the
> nominal design condition, and **beats** it under actuator failure, surface
> jam, and off-design flight condition — while using a fixed communication
> budget independent of the number of surfaces.

Robustness is where a distributed learned policy should genuinely win, because
centralised LQG is synthesised for one plant and one actuator set. That is the
thesis, and it is falsifiable.

### Scaling / emergence study

Three surfaces may be too few for "emergence" to mean much. The plant already
accepts an arbitrary surface count, so run 3 / 6 / 10 / 16 surfaces and show the
phase-gradient pattern sharpening and the communication cost staying flat.

---

## 5. Honest risks and limitations

| Risk | Mitigation |
|---|---|
| `r_k^phys` needs the modal control influence `v_r^T b_k`. Free in simulation; on hardware requires a modal model. | Sweep 10/20/30% error in `B` and show credit quality degrades gracefully. State it as a limitation. |
| Phase-locked actions assume near-harmonic response; turbulence is broadband. | The additive broadband channel is part of the design, not a patch. Ablate it. |
| Proposition 2 is a truncation bound, not an exact result. | State it as such. Do not oversell. |
| Quasi-steady flap aero over-estimates high-frequency control authority. | Documented in the plant module. Re-validate the final policy on SHARPy (UVLM + nonlinear beam) as the high-fidelity check. |
| "Emergent" is a loaded word. | Define it operationally as the spanwise phase gradient, and measure it. Never claim more. |

---

## 6. Naming

- **MoCCA** — Modal Consensus & Credit Assignment. Current working name.
- **HARMONY** — alternative if a more evocative name is wanted; fits a paper
  about distributed agents agreeing on phase.

AEGIS stays the project/repository name; the algorithm gets its own.
