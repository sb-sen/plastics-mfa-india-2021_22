import math

import numpy as np
from scipy.stats import norm

N_ITERATIONS = 10000
BASE_SEED = 42

A_COEF = 0.75
B_COEF = 1.105
K_LIN = 6.88
K_LIN_RELIABILITY = 20.02
D_LIN_RELIABILITY = 2.26

SCORES = {
    "W": (2, 2, 1, 1, 2),
    "c": (2, 2, 1, 1, 3),
    "phi": (3, 3, 2, 2, 3),
    "eta": (2, 3, 2, 2, 2),
    "tau": (3, 3, 2, 2, 3),
    "a_rec": (2, 3, 1, 2, 3),
    "a_en": (2, 3, 1, 2, 3),
    "a_reu": (2, 3, 1, 2, 3),
}

MODES = {
    "W": 3921047.77,
    "c": 0.9162,
    "phi": 0.20,
    "eta": 0.70,
    "tau": 0.17,
    "a_rec": 0.32,
    "a_en": 0.1425,
    "a_reu": 0.02,
}

OBSERVED_RANGE = {"eta": (0.546, 0.781), "tau": (0.0, 0.20)}

LABELS = {
    "W": "Urban plastics waste (t)",
    "c": "Collection efficiency",
    "phi": "Informal share",
    "eta": "Informal recovery fraction",
    "tau": "Reutilised share of rejects",
    "a_rec": "Formal recycled fraction",
    "a_en": "Formal energy-related fraction",
    "a_reu": "Formal reutilised fraction",
}

FLOWS = [
    ("Urban plastics waste", "waste"),
    ("Urban collected", "collected"),
    ("Urban uncollected", "uncollected"),
    ("Urban formal", "formal"),
    ("Urban informal", "informal"),
    ("Formal recycled", "formal_recycled"),
    ("Formal energy recovery", "formal_energy"),
    ("Formal reutilised", "formal_reutilised"),
    ("Formal landfills and dumpsites", "formal_landfilled"),
    ("Informal recycled", "informal_recycled"),
    ("Informal rejects", "informal_rejects"),
    ("Rejects reutilised", "rejects_reutilised"),
    ("Rejects not reutilised", "rejects_not_reutilised"),
    ("Recycled", "recycled"),
    ("Reutilised", "reutilised"),
    ("Landfills and dumpsites", "landfilled"),
    ("Energy recovery", "energy"),
    ("Uncollected waste and informal rejects", "uncollected_rejects"),
    ("CPI", "CPI"),
    ("LDS", "LDS"),
]
FLOW_KEY = dict(FLOWS)

MAIN_OUTPUTS = [
    "Recycled",
    "Reutilised",
    "Energy recovery",
    "Landfills and dumpsites",
    "Uncollected waste and informal rejects",
    "CPI",
    "LDS",
]


def cv_exponential(score, reliability=False):
    if reliability:
        return A_COEF * math.exp(B_COEF * score)
    if score == 1:
        return 0.0
    return A_COEF * math.exp(B_COEF * (score - 1))


def cv_linear(score, reliability=False):
    if reliability:
        return D_LIN_RELIABILITY + K_LIN_RELIABILITY * (score - 1)
    return K_LIN * (score - 1)


def total_cv(scores, kind):
    cv = cv_exponential if kind == "exp" else cv_linear
    reliability, completeness, temporal, geographical, other = scores
    parts = [
        cv(reliability, True),
        cv(completeness),
        cv(temporal),
        cv(geographical),
        cv(other),
    ]
    return math.sqrt(sum(p * p for p in parts)) / 100.0


def parameter_bounds(key, kind):
    mode = MODES[key]
    half_width = math.sqrt(6.0) * total_cv(SCORES[key], kind) * mode
    low, high = mode - half_width, mode + half_width
    if key != "W":
        low, high = max(low, 0.0), min(high, 1.0)
    if key in OBSERVED_RANGE:
        low = min(low, OBSERVED_RANGE[key][0])
        high = max(high, OBSERVED_RANGE[key][1])
    return low, high


def build_config(kind):
    config = {}
    for key in MODES:
        low, high = parameter_bounds(key, kind)
        config[key] = (low, MODES[key], high)
    return config


CONFIG = {"exp": build_config("exp"), "lin": build_config("lin")}


def sample_inputs(config, n, rng):
    return {key: rng.triangular(low, mode, high, n) for key, (low, mode, high) in config.items()}


def impose_rank_correlation(columns, correlation, rng):
    n, k = columns.shape
    scores = norm.ppf(np.arange(1, n + 1) / (n + 1))
    shuffled = np.column_stack([rng.permutation(scores) for _ in range(k)])
    shuffled_corr = np.corrcoef(shuffled, rowvar=False)
    transformed = (
        shuffled
        @ np.linalg.inv(np.linalg.cholesky(shuffled_corr)).T
        @ np.linalg.cholesky(correlation).T
    )
    result = np.empty_like(columns)
    for j in range(k):
        ranks = np.argsort(np.argsort(transformed[:, j]))
        result[:, j] = np.sort(columns[:, j])[ranks]
    return result


def apply_dependence(inputs, keys, rho, rng):
    k = len(keys)
    correlation = np.full((k, k), rho)
    np.fill_diagonal(correlation, 1.0)
    reordered = impose_rank_correlation(
        np.column_stack([inputs[key] for key in keys]), correlation, rng
    )
    result = dict(inputs)
    for j, key in enumerate(keys):
        result[key] = reordered[:, j]
    return result


def run_model(inputs):
    n = len(inputs["W"])
    c, phi, eta, tau = inputs["c"], inputs["phi"], inputs["eta"], inputs["tau"]
    a_rec, a_reu, a_en = inputs["a_rec"], inputs["a_reu"], inputs["a_en"]
    a_landfill = 1.0 - (a_rec + a_reu + a_en)
    if not (a_landfill > 0).all():
        raise ValueError("formal landfill and dumpsite fraction became negative")

    transfer = np.zeros((n, 10, 10))
    transfer[:, 1, 0] = c
    transfer[:, 9, 0] = 1 - c
    transfer[:, 2, 1] = 1 - phi
    transfer[:, 3, 1] = phi
    transfer[:, 5, 2] = a_rec
    transfer[:, 6, 2] = a_reu
    transfer[:, 7, 2] = a_en
    transfer[:, 8, 2] = a_landfill
    transfer[:, 5, 3] = eta
    transfer[:, 4, 3] = 1 - eta
    transfer[:, 6, 4] = tau
    transfer[:, 9, 4] = 1 - tau

    system = np.eye(10)[None, :, :] - transfer
    supply = np.zeros((n, 10, 1))
    supply[:, 0, 0] = inputs["W"]
    x = np.linalg.solve(system, supply)[:, :, 0]
    waste = inputs["W"]

    out = {
        "waste": waste,
        "collected": x[:, 1],
        "uncollected": waste - x[:, 1],
        "formal": x[:, 2],
        "informal": x[:, 3],
        "formal_recycled": x[:, 2] * a_rec,
        "formal_energy": x[:, 2] * a_en,
        "formal_reutilised": x[:, 2] * a_reu,
        "formal_landfilled": x[:, 2] * a_landfill,
        "informal_recycled": x[:, 3] * eta,
        "informal_rejects": x[:, 4],
        "rejects_reutilised": x[:, 4] * tau,
        "rejects_not_reutilised": x[:, 4] * (1 - tau),
        "recycled": x[:, 5],
        "reutilised": x[:, 6],
        "energy": x[:, 7],
        "landfilled": x[:, 8],
        "uncollected_rejects": x[:, 9],
    }
    out["CPI"] = (out["recycled"] + out["reutilised"]) / waste
    out["LDS"] = (out["landfilled"] + out["uncollected_rejects"]) / waste

    if not np.allclose(x[:, 5:10].sum(axis=1), waste, rtol=1e-9):
        raise ValueError("mass balance violated")
    chain = waste * c * ((1 - phi) * a_rec + phi * eta)
    if not np.allclose(chain, out["recycled"], rtol=1e-9):
        raise ValueError("matrix solution does not match the flow chain")
    if not np.allclose(out["CPI"] + out["LDS"] + out["energy"] / waste, 1.0, rtol=1e-9):
        raise ValueError("indicators and energy recovery do not sum to one")
    return out


def simulate(config, n=N_ITERATIONS, seed=BASE_SEED):
    rng = np.random.default_rng(seed)
    inputs = sample_inputs(config, n, rng)
    return inputs, run_model(inputs)


def summary_statistics(values):
    return {
        "Mean": values.mean(),
        "Median": np.median(values),
        "CI95_low": np.percentile(values, 2.5),
        "CI95_high": np.percentile(values, 97.5),
    }


def interval_width(values):
    return np.diff(np.percentile(values, [2.5, 97.5]))[0]
