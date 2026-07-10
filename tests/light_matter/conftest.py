"""Shared fixtures for the light_matter test suite.

The BIE sweep is the expensive computation; these module-scoped fixtures build
the ``circle_weak`` solver and spectrum once and let every test in a module
reuse them. The spectrum fixture depends on the solver fixture so the BIE work
happens exactly once.
"""

from __future__ import annotations

import pytest

from qubit_playground.light_matter.environment import (
    EnvironmentSpectrum,
    compute_spectrum,
)
from qubit_playground.light_matter.scenarios import (
    ScenarioConfig,
    build_solver,
    load_scenario,
)


@pytest.fixture(scope="module")
def circle_weak_config() -> ScenarioConfig:
    return load_scenario("circle_weak")


@pytest.fixture(scope="module")
def circle_weak_solver(circle_weak_config: ScenarioConfig):
    return build_solver(circle_weak_config)


@pytest.fixture(scope="module")
def circle_weak_spectrum(
    circle_weak_config: ScenarioConfig, circle_weak_solver
) -> EnvironmentSpectrum:
    import numpy as np

    sweep = circle_weak_config.sweep
    wavelengths = np.linspace(sweep.lambda_min_nm, sweep.lambda_max_nm, sweep.n_points)
    return compute_spectrum(
        circle_weak_solver,
        wavelengths,
        circle_weak_config.emitter.x_s,
        circle_weak_config.emitter.z_s,
    )
