# Probabilistic material flow analysis of municipal plastics waste in India

Vinod Gupta School of Management, Indian Institute of Technology Kharagpur

Code for the circularity assessment of urban plastics waste in India (reference year 2021-22) using probabilistic material flow analysis (MFA), submitted to Waste Management \& Research (manuscript WMR-26-0470).

## What the code does

1. Converts data-quality scores (reliability, completeness, temporal, geographical and other correlation) into coefficients of variation, using the exponential and linear functions (Laner et al. 2016), and derives triangular input ranges.
2. Propagates the input uncertainty through the material flow model by Monte Carlo simulation (10,000 iterations, seed 42).
3. Computes the Circularity Performance Index (CPI) and the Landfill and Dumpsite Share (LDS).
4. Tests the effect of rank correlations between inputs (Iman-Conover), the convergence of the simulation (running mean, 20 seeds, 100,000 iterations) and the sensitivity of the outputs (Spearman rank correlation).

## Files

|File|Content|
|-|-|
|`mfa\\\_model.py`|Data-quality scheme, input ranges, sampling, rank-correlation method, material flow model|
|`run\\\_analysis.py`|Runs the base and linear-type uncertainty analyses, the dependence tests, the convergence tests and the sensitivity analysis, and writes tables and figures to `outputs/`|
|`tests/test\\\_model.py`|Checks the mass balance, the indicators and the reported base-run means|
|`outputs/`|Tables and figures produced by `run\\\_analysis.py`|

## 

Usage
pip install -r requirements.txt
python run\\\_analysis.py
pytest
Use `--out <folder>` to change the output folder and `--skip-convergence` to omit the 20-seed and 100,000-iteration runs.

## Model

Nodes: generation, collected, formal, informal, informal rejects, recycled, reutilised, energy recovery, landfills and dumpsites, and uncollected waste and informal rejects. 

* CPI = (recycled + reutilised) / urban plastics waste
* LDS = (landfills and dumpsites + uncollected waste and informal rejects) / urban plastics waste

## Input parameters

The modes, data-quality scores and survey ranges are set at the top of `mfa\\\_model.py` and are documented in the Supplementary Text of the article.

## Reproducibility

Random numbers come from `numpy.random.default\\\_rng` with fixed seeds (42 for the base run, 1001-1020 for the seed test). The outputs were produced with the package versions in `requirements.txt`.

## Citation

Please cite the article (details to be added if accepted) and this repository (see `CITATION.cff`).

## License

See the `LICENSE` file.

