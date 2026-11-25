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


def test_scenarios_mass_balance_and_prevention():
    from scenarios import END_OF_LIFE, simulate_scenarios

    results = simulate_scenarios(CONFIG["exp"], n=2000, seed=11)
    base = results["Baseline"]
    for name, out in results.items():
        total = sum(out[key] for key in END_OF_LIFE)
        assert np.allclose(total, out["waste"]), name
        assert np.allclose(out["CPI"] + out["LDS"] + out["energy"] / out["waste"], 1.0), name
    for name, out in results.items():
        assert all((out[key] >= -1e-6).all() for key in END_OF_LIFE), name
    assert np.allclose(results["A1"]["CPI"], base["CPI"])
    assert np.allclose(results["A1"]["LDS"], base["LDS"])
    assert (results["A2"]["CPI"] > base["CPI"]).all()
    assert (results["B (zeta=1)"]["recycled"] >= base["recycled"] - 1e-6).all()
    assert np.allclose(results["C (d=1, chi=0.5)"]["landfilled"], 0.0, atol=1e-6)
    assert results["D2"]["CPI"].mean() > results["D1"]["CPI"].mean() > base["CPI"].mean()


def test_baseline_unchanged_in_scenario_run():
    from scenarios import simulate_scenarios

    results = simulate_scenarios(CONFIG["exp"])
    assert abs(results["Baseline"]["recycled"].mean() - 1351179) < 1
