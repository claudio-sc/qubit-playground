import pytest

from qubit_playground.lossy_oscillator import DecayResult, simulate_photon_decay


@pytest.fixture(scope="module")
def result() -> DecayResult:
    return simulate_photon_decay()  # defaults: dim=16, kappa=0.2, alpha_0=2
