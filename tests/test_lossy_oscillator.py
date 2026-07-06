import jax.numpy as jnp
import numpy as np

from qubit_playground.lossy_oscillator import (
    DecayResult,
    max_absolute_error,
    simulate_photon_decay,
)


def test_matches_analytic_curve(result: DecayResult) -> None:
    assert max_absolute_error(result) < 1e-4


def test_fitted_decay_rate(result: DecayResult) -> None:
    times = np.asarray(result.times, dtype=np.float64)
    photon_number = np.asarray(result.photon_number, dtype=np.float64)
    slope, _ = np.polyfit(times, np.log(photon_number), 1)
    assert np.isclose(slope, -result.kappa, rtol=1e-3)


def test_truncation_error_grows_when_dim_too_small() -> None:
    error_dim8 = max_absolute_error(simulate_photon_decay(dim=8))
    error_dim16 = max_absolute_error(simulate_photon_decay(dim=16))
    assert error_dim8 > 10 * error_dim16


def test_error_metric_is_zero_on_identical_curves() -> None:
    curve = jnp.array([1.0, 2.0, 3.0])
    identical = DecayResult(
        times=jnp.array([0.0, 1.0, 2.0]),
        photon_number=curve,
        analytic=curve,
        kappa=0.2,
    )
    assert max_absolute_error(identical) == 0.0
