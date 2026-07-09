"""Light-matter interaction: coupling a Maxwell solver to the master equation.

Implements the physics of Bouchet & Carminati, JOSA A 36, 186 (2019): a driven
three-level emitter (optical Bloch equations), the weak-coupling Purcell effect,
and strong coupling / vacuum Rabi splitting. The classical environment response
S(r_s, r_s, omega) is computed by the boundary-integral solver pysie2d (NumPy);
the quantum master equations are solved with dynamiqs (JAX). Data flows one way:
pysie2d -> plain floats/arrays -> dynamiqs.

See Part A of the code-spec for the unit system and physics conventions.
"""

from __future__ import annotations

from qubit_playground.light_matter.bloch import (
    BlochResult,
    simulate_driven_emitter,
    steady_state,
)
from qubit_playground.light_matter.emitter import ThreeLevelEmitter
from qubit_playground.light_matter.environment import (
    EnvironmentSpectrum,
    PhotonicMode,
    compute_spectrum,
)
from qubit_playground.light_matter.fitting import (
    LorentzianFit,
    fit_complex_exponentials,
    fit_lorentzian,
)
from qubit_playground.light_matter.purcell import (
    PurcellResult,
    simulate_modified_decay,
)
from qubit_playground.light_matter.scenarios import (
    ScenarioConfig,
    build_solver,
    compute_scenario_spectrum,
    list_scenarios,
    load_scenario,
)
from qubit_playground.light_matter.strong_coupling import (
    StrongCouplingResult,
    classical_eigenfrequencies,
    simulate_jaynes_cummings,
)
from qubit_playground.light_matter.units import (
    C_NM_PER_FS,
    omega_to_wavelength,
    wavelength_to_omega,
)

__all__ = [
    "C_NM_PER_FS",
    "BlochResult",
    "EnvironmentSpectrum",
    "LorentzianFit",
    "PhotonicMode",
    "PurcellResult",
    "ScenarioConfig",
    "StrongCouplingResult",
    "ThreeLevelEmitter",
    "build_solver",
    "classical_eigenfrequencies",
    "compute_scenario_spectrum",
    "compute_spectrum",
    "fit_complex_exponentials",
    "fit_lorentzian",
    "list_scenarios",
    "load_scenario",
    "omega_to_wavelength",
    "simulate_driven_emitter",
    "simulate_jaynes_cummings",
    "simulate_modified_decay",
    "steady_state",
    "wavelength_to_omega",
]
