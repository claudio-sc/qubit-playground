"""Strong coupling / vacuum Rabi splitting -- the flagship (paper Sec. 4C).

The headline claim: a classical boundary-integral Maxwell solver predicts the
eigenfrequencies of a strongly coupled emitter-resonator system, and an
independent quantum master-equation simulation reproduces them.

Classical side (Part A.4, paper Eq. (37)): the two roots of
    (omega_0' - varpi - i*gamma_0'/2)(omega_m - varpi - i*gamma_m/2) = g^2
    varpi_pm = (varpi_0' + varpi_m)/2
               +- sqrt(((varpi_m - varpi_0')/2)^2 + g^2)
with varpi_0' = omega_0' - i*gamma_0'/2 and varpi_m = omega_m - i*gamma_m/2.

Quantum side: a Jaynes-Cummings simulation in the frame rotating at omega_0',
    H = Delta_m a†a + g(a†sigma_- + a sigma_+),  Delta_m = omega_m - omega_0'
    jumps sqrt(gamma_m) a, sqrt(gamma_0') sigma_-.
In the <=1-excitation manifold the equations for <sigma_-> and <a> close exactly
on the 2x2 non-Hermitian matrix whose eigenvalues are varpi_pm - omega_0', so the
quantum-vs-classical comparison is exact in principle (limited only by the pencil
extraction), not approximate.

Everything needed is inside the ``PhotonicMode`` argument (mode, coupling, and
dressed emitter), so these functions take only a mode -- removing the class of
bugs where a caller fits with one emitter and simulates with another.
"""

from __future__ import annotations

from dataclasses import dataclass

import dynamiqs as dq
import numpy as np
from jax import Array

from qubit_playground.light_matter.environment import PhotonicMode
from qubit_playground.light_matter.fitting import fit_complex_exponentials

dq.set_precision("double")


@dataclass(frozen=True)
class StrongCouplingResult:
    """Outcome of a Jaynes-Cummings strong-coupling simulation.

    Attributes:
        times: Save times (fs).
        qubit_population: <sigma_+ sigma_-> vs time (real).
        cavity_population: <a† a> vs time (real).
        coherence_qubit: <sigma_-> vs time (complex).
        coherence_cavity: <a> vs time (complex).
        poles_quantum: Lab-frame poles from the pencil, sorted by real part.
        poles_classical: Classical eigenfrequencies, same sorting.
        mode: The PhotonicMode driving the simulation.
    """

    times: Array
    qubit_population: Array
    cavity_population: Array
    coherence_qubit: Array
    coherence_cavity: Array
    poles_quantum: tuple[complex, complex]
    poles_classical: tuple[complex, complex]
    mode: PhotonicMode


def classical_eigenfrequencies(mode: PhotonicMode) -> tuple[complex, complex]:
    """Classical polariton eigenfrequencies varpi_pm (paper Eq. (37)).

    Args:
        mode: The photonic mode (carries omega_m, gamma_m, g, and the dressed
            emitter parameters).

    Returns:
        The two complex roots (varpi_-, varpi_+), sorted by real part.
    """
    varpi_0p = mode.omega_emitter_dressed - 1j * mode.gamma_emitter_dressed / 2.0
    varpi_m = mode.omega_m - 1j * mode.gamma_m / 2.0
    disc = np.sqrt(((varpi_m - varpi_0p) / 2.0) ** 2 + mode.g**2)  # Eq. (37)
    roots = [(varpi_0p + varpi_m) / 2.0 - disc, (varpi_0p + varpi_m) / 2.0 + disc]
    roots.sort(key=lambda z: z.real)
    return complex(roots[0]), complex(roots[1])


def simulate_jaynes_cummings(
    mode: PhotonicMode,
    *,
    t_final: float,
    n_times: int = 600,
) -> StrongCouplingResult:
    """Simulate the Jaynes-Cummings dynamics and extract the polariton poles.

    Hilbert space is qubit (x) cavity with ``n_fock = 3`` (dynamics is confined
    to <=1 excitation from the initial state; 3 leaves headroom at no cost). The
    frame rotates at omega_0' = ``mode.omega_emitter_dressed``. The initial state
    (|a>+|b>)/sqrt(2) (x) |0> populates the coherences carrying both poles.

    Poles are extracted by a matrix pencil (n_modes=2) on <sigma_-(t)>, then
    shifted back to the lab frame by adding omega_0' (real, so only the real
    parts shift; decay rates are frame-independent).

    Args:
        mode: The photonic mode to simulate.
        t_final: Final simulation time (fs).
        n_times: Number of save points.

    Returns:
        A StrongCouplingResult with quantum and classical poles.

    Raises:
        ValueError: If the time step undersamples the dynamics (the pencil would
            then return aliased, plausible-looking but wrong poles).
    """
    n_fock = 3
    g = mode.g
    gamma_m = mode.gamma_m
    gamma_0p = mode.gamma_emitter_dressed
    omega_0p = mode.omega_emitter_dressed
    delta_m = mode.omega_m - omega_0p

    # qubit (x) cavity operators (qubit ground |a>=|0>, excited |b>=|1>). Build
    # in dense layout so the mixed sparse/dense Hamiltonian sum needs no implicit
    # conversion (which otherwise warns).
    ket_g = dq.basis(2, 0)
    ket_e = dq.basis(2, 1)
    sigma_minus_q = ket_g @ dq.dag(ket_e)  # |a><b|
    eye_cav = dq.eye(n_fock, layout=dq.dense)
    eye_q = dq.eye(2, layout=dq.dense)
    cav = dq.destroy(n_fock, layout=dq.dense)

    sm = dq.tensor(sigma_minus_q, eye_cav)
    sp = dq.dag(sm)
    a = dq.tensor(eye_q, cav)
    a_dag = dq.dag(a)

    hamiltonian = delta_m * (a_dag @ a) + g * (a_dag @ sm + a @ sp)
    jump_ops = []
    if gamma_m > 0.0:
        jump_ops.append(np.sqrt(gamma_m) * a)
    if gamma_0p > 0.0:
        jump_ops.append(np.sqrt(gamma_0p) * sm)

    psi0 = dq.unit(dq.tensor(ket_g + ket_e, dq.basis(n_fock, 0)))
    tsave = np.linspace(0.0, t_final, n_times)

    result = dq.mesolve(
        hamiltonian,
        jump_ops,
        psi0,
        tsave,
        exp_ops=[sp @ sm, a_dag @ a, sm, a],
        method=dq.method.Tsit5(rtol=1e-10, atol=1e-12),
    )

    dt = float(tsave[1] - tsave[0])
    # Sampling guard: aliasing would make the pencil return wrong-but-plausible
    # poles, so refuse to fit an undersampled signal.
    nyquist = np.pi / (abs(delta_m) + 2.0 * g + max(gamma_m, gamma_0p))
    if dt >= nyquist:
        raise ValueError(
            f"time step dt={dt:.4g} undersamples the dynamics "
            f"(need dt < {nyquist:.4g}); increase n_times or shorten t_final"
        )

    coherence_qubit = np.asarray(result.expects[2])
    poles_rotating = fit_complex_exponentials(coherence_qubit, dt, n_modes=2)
    poles_lab = poles_rotating + omega_0p  # real shift only
    poles_lab = poles_lab[np.argsort(poles_lab.real)]
    poles_quantum = (complex(poles_lab[0]), complex(poles_lab[1]))

    return StrongCouplingResult(
        times=tsave,
        qubit_population=np.real(result.expects[0]),
        cavity_population=np.real(result.expects[1]),
        coherence_qubit=coherence_qubit,
        coherence_cavity=np.asarray(result.expects[3]),
        poles_quantum=poles_quantum,
        poles_classical=classical_eigenfrequencies(mode),
        mode=mode,
    )
