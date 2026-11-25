# Circularity assessment of plastics in municipal solid waste in India using material flow analysis

Vinod Gupta School of Management, Indian Institute of Technology Kharagpur

Model code for the circularity assessment of plastics in municipal solid waste in India (reference year 2021-22) using probabilistic material flow analysis (MFA), submitted to Waste Management & Research (manuscript WMR-26-0470). Input data, data-quality scores and all results are documented in the article and its Supplementary Information; this repository contains the code for the analysis.

## What the code does

- Converts data-quality scores (reliability, completeness, temporal, geographical and other correlation) into coefficients of variation, using the exponential and linear functions of Laner et al. (2016), and derives triangular input ranges.
- Propagates the input uncertainty through the material flow model (matrix form, A x = b, with A = I - T) by Monte Carlo simulation (10,000 iterations, seed 42).
- Computes the Circularity Performance Index (CPI) and the Landfill and Dumpsite Share (LDS).
- Evaluates policy scenarios (ban on single-use plastics, extended producer responsibility for packaging, landfill diversion and their combination) on the same random draws as the baseline.
- Tests the effect of rank correlations between inputs (Iman-Conover), the convergence of the simulation (running mean, 20 seeds, 100,000 iterations) and the sensitivity of the outputs (Spearman rank correlation).

## Files

| File | Content |
|---|---|
| `mfa_model.py` | Data-quality scheme, input ranges, sampling, rank-correlation method, material flow model |
| `scenarios.py` | Scenario parameters (prevention, EPR, landfill diversion, combined scenarios) |
| `run_analysis.py` | Runs the base and linear-type uncertainty analyses, the dependence tests, the convergence tests and the sensitivity analysis |
| `run_scenarios.py` | Runs the scenarios with the exponential-type and linear-type ranges |
| `tests/test_model.py` | Checks the mass balance, the indicators and the reported base-run means |

## Usage

```
pip install -r requirements.txt
python run_analysis.py
python run_scenarios.py
pytest
```

Results (tables and figures) are written to an `outputs/` folder, created when the scripts are run. Use `--out <folder>` to change it, and `--skip-convergence` with `run_analysis.py` to omit the 20-seed and 100,000-iteration runs. The full analysis takes a few seconds on a laptop (a few minutes with the convergence tests).


## Input parameters

The modes, data-quality scores and survey ranges are set at the top of `mfa_model.py` and are documented in the Supplementary Information of the article.

## Random seeds and package versions

Random numbers come from `numpy.random.default_rng` with fixed seeds (42 for the base run, 43 for the prevention-rate draws in the scenarios, 1001-1020 for the seed test). The results in the article were produced with the package versions in `requirements.txt`.

## Citation

Please cite the article (details to be added on publication) and this repository (see `CITATION.cff`).

## License

MIT, see the `LICENSE` file.
