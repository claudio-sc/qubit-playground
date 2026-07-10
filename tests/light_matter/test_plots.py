from pathlib import Path

import numpy as np


def test_plot_bloch_writes_png(tmp_path: Path) -> None:
    from qubit_playground.light_matter.plot_bloch import plot_steady_state

    delta = np.linspace(-3.0, 3.0, 10)
    analytic = np.tile(0.1 / (1.0 + delta**2), (2, 1))
    out = tmp_path / "obe.png"
    plot_steady_state(delta, analytic, delta, analytic, ["a", "b"], out)
    assert out.stat().st_size > 0


def test_plot_purcell_writes_png(tmp_path: Path) -> None:
    from qubit_playground.light_matter.plot_purcell import plot_purcell

    ldos = np.full((5, 5), 1.2)
    ldos[2, 2] = np.nan
    times = np.linspace(0.0, 10.0, 20)
    curve = 0.5 * np.exp(-0.1 * times)
    out = tmp_path / "purcell.png"
    plot_purcell(
        ldos,
        (-1.0, 1.0, -1.0, 1.0),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 0.0, -1.0]),
        (0.0, 0.5),
        times,
        curve,
        curve,
        curve,
        curve,
        out,
    )
    assert out.stat().st_size > 0


def test_plot_strong_coupling_writes_png(tmp_path: Path) -> None:
    from qubit_playground.light_matter.plot_strong_coupling import plot_vacuum_rabi

    omega = np.linspace(2.6, 2.8, 10)
    im_s = 1.0 / (1.0 + ((omega - 2.7) / 0.01) ** 2)
    g0 = np.logspace(-4, -2, 8)
    re_c = np.vstack([np.full_like(g0, 2.69), np.full_like(g0, 2.71)])
    im_c = np.vstack([np.full_like(g0, -0.002), np.full_like(g0, -0.002)])
    g0m = g0[::2]
    re_q = re_c[:, ::2]
    out = tmp_path / "rabi.png"
    plot_vacuum_rabi(omega, im_s, omega, im_s, g0, re_c, im_c, g0m, re_q, out)
    assert out.stat().st_size > 0
