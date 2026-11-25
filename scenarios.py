import numpy as np

from mfa_model import BASE_SEED, N_ITERATIONS, run_model, sample_inputs

PSI = 0.89
OMEGA = 0.72
R_RIGID = 0.80
R_FLEXIBLE = 0.60
R_PACK = OMEGA * R_FLEXIBLE + (1.0 - OMEGA) * R_RIGID
ZETA_PARTIAL = 0.83
BETA_RANGE = (0.05, 0.10, 0.15)

SCENARIOS = {
    "A1": {"lam": 0.0},
    "A2": {"lam": 1.0},
    "B (zeta=1)": {"zeta": 1.0},
    "B (zeta=0.83)": {"zeta": ZETA_PARTIAL},
    "C (d=0.5, chi=0)": {"d": 0.5, "chi": 0.0},
    "C (d=0.5, chi=0.5)": {"d": 0.5, "chi": 0.5},
    "C (d=0.5, chi=1)": {"d": 0.5, "chi": 1.0},
    "C (d=1, chi=0)": {"d": 1.0, "chi": 0.0},
    "C (d=1, chi=0.5)": {"d": 1.0, "chi": 0.5},
    "C (d=1, chi=1)": {"d": 1.0, "chi": 1.0},
    "D1": {"lam": 0.0, "zeta": ZETA_PARTIAL, "d": 0.5, "chi": 0.5},
    "D2": {"lam": 0.0, "zeta": 1.0, "d": 1.0, "chi": 1.0},
}

END_OF_LIFE = ["recycled", "reutilised", "energy", "landfilled", "uncollected_rejects"]
NON_RETAINING = ["landfilled", "uncollected_rejects"]


def apply_epr(inputs, zeta):
    a_rec, a_reu, a_en = inputs["a_rec"], inputs["a_reu"], inputs["a_en"]
    a_lan = 1.0 - (a_rec + a_reu + a_en)
    target = (1.0 - PSI) * a_rec + PSI * zeta * R_PACK
    new_rec = np.minimum(a_rec + a_lan, np.maximum(a_rec, target))
    return {**inputs, "a_rec": new_rec}


def apply_diversion(inputs, d, chi):
    a_rec, a_reu, a_en = inputs["a_rec"], inputs["a_reu"], inputs["a_en"]
    a_lan = 1.0 - (a_rec + a_reu + a_en)
    diverted = d * a_lan
    return {**inputs, "a_rec": a_rec + chi * diverted, "a_en": a_en + (1.0 - chi) * diverted}


def apply_prevention(out, beta, lam):
    waste = out["waste"]
    nr_total = out["landfilled"] + out["uncollected_rejects"]
    lam = np.minimum(lam, nr_total * (1.0 - beta) / (beta * (waste - nr_total)))
    keep = 1.0 - (1.0 - lam) * beta
    new = {"waste": waste * (1.0 - beta), "lam_effective": lam}
    for key in END_OF_LIFE:
        flow = keep * out[key]
        if key in NON_RETAINING:
            flow = flow - lam * beta * waste * out[key] / nr_total
        new[key] = flow
    new["CPI"] = (new["recycled"] + new["reutilised"]) / new["waste"]
    new["LDS"] = (new["landfilled"] + new["uncollected_rejects"]) / new["waste"]
    total = sum(new[key] for key in END_OF_LIFE)
    if not np.allclose(total, new["waste"], rtol=1e-9):
        raise ValueError("mass balance violated after prevention")
    return new


def run_scenario(inputs, beta, params):
    scenario_inputs = inputs
    if "zeta" in params:
        scenario_inputs = apply_epr(scenario_inputs, params["zeta"])
    if "d" in params:
        scenario_inputs = apply_diversion(scenario_inputs, params["d"], params["chi"])
    out = run_model(scenario_inputs)
    if "lam" in params:
        out = apply_prevention(out, beta, params["lam"])
    return out


def simulate_scenarios(config, n=N_ITERATIONS, seed=BASE_SEED):
    rng = np.random.default_rng(seed)
    inputs = sample_inputs(config, n, rng)
    beta = np.random.default_rng(seed + 1).triangular(*BETA_RANGE, n)
    results = {"Baseline": run_model(inputs)}
    for name, params in SCENARIOS.items():
        results[name] = run_scenario(inputs, beta, params)
    return results
