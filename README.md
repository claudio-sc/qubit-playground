# qubit-playground

![ci](https://github.com/claudio-sc/qubit-playground/actions/workflows/ci.yml/badge.svg)

Hands-on studies of differentiable open-quantum-system simulation with
[dynamiqs](https://github.com/dynamiqs/dynamiqs)/[JAX](https://github.com/jax-ml/jax),
motivated by superconducting cat-qubit platforms. Current content: a validated
lossy harmonic oscillator; see [Roadmap](#roadmap).

## The physics example — lossy harmonic oscillator

A single cavity mode with Hamiltonian `H = ω a†a` under single-photon loss
(jump operator `√κ a`) obeys the Lindblad master equation. Starting from a
coherent state `|α₀⟩`, the mean photon number decays purely exponentially:

```
⟨a†a⟩(t) = |α₀|² · exp(−κ·t)
```

![Photon-number decay](https://raw.githubusercontent.com/claudio-sc/qubit-playground/v0.1.0/figures/lossy_oscillator_decay.png)

The dynamiqs simulation (`dq.mesolve`) reproduces this to within a maximum
absolute error of ~5e-6 over the full time window. This agreement is enforced
by the test suite in CI.

## Setup

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```bash
uv sync                                                    # create env, install deps
uv run python -m qubit_playground.plot_lossy_oscillator   # regenerate the figure
uv run pytest                                              # run the test suite
```

## Layout

```
src/qubit_playground/
  lossy_oscillator.py       # model + simulation, analytic reference, error metric
  plot_lossy_oscillator.py  # figure generation + validation report
tests/                      # analytic, decay-rate, truncation, and plot tests
figures/                    # generated plots
```

## Roadmap

- Fock-space and cat-state studies.
- Notes on solver internals (Diffrax).
- A semiclassical light–matter module driven by an external 2-D EM solver.
