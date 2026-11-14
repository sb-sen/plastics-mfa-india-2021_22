import numpy as np

from mfa_model import CONFIG, FLOW_KEY, simulate


def test_base_run_reproduces_reported_means():
    _, out = simulate(CONFIG["exp"])
    assert abs(out["recycled"].mean() - 1351179) < 1
    assert abs(out["landfilled"].mean() - 1411602) < 1
    assert abs(out["uncollected_rejects"].mean() - 682412) < 1
    assert abs(out["CPI"].mean() - 0.3664) < 1e-4
    assert abs(out["LDS"].mean() - 0.5344) < 1e-4


def test_mass_balance_and_indicators():
    _, out = simulate(CONFIG["lin"], n=2000, seed=7)
    sinks = (
        out["recycled"]
        + out["reutilised"]
        + out["energy"]
        + out["landfilled"]
        + out["uncollected_rejects"]
    )
    assert np.allclose(sinks, out["waste"])
    assert np.allclose(out["CPI"] + out["LDS"] + out["energy"] / out["waste"], 1.0)


def test_flow_keys_exist():
    _, out = simulate(CONFIG["exp"], n=100, seed=1)
    assert all(key in out for key in FLOW_KEY.values())
