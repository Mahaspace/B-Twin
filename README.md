# B-Twin: Causal Inference on Time Series with Hidden Confounding

Official implementation of **B-Twin**, a representation-learning framework for estimating treatment effects from time series in the presence of latent confounding.

B-Twin combines latent propensity score balancing and synthetic-control-style counterfactual construction. The method learns treatment-relevant representations from pre-treatment trajectories and uses them to identify comparable control units for treatment effect estimation.

📄 **Paper:** *Balanced Twins: Causal Inference on Time Series with Hidden Confounding*  
**arXiv:** [(https://arxiv.org/abs/2606.18969)]

---

## Repository Contents

This repository contains:

- The proposed **B-Twin** framework and its variants.
- Classical causal inference baselines.
- Neural causal inference baselines.
- The SyncTwin implementation.
- Simulation environments used in the paper.
- Scripts for reproducing experiments on simulated and real-world datasets.

---

## Project Structure

```text
.
├── run_simulations.py
├── run_on_real_data.py
├── methods/
│   ├── classic_baselines/
│   ├── proposed_model/
│   ├── baselines/
│   ├── Synctwin/
│   └── simulations/
├── data/
├── results/
├── figures/
├── notebook/
└── pyproject.toml
```

---

## Installation

Install all dependencies using:

```bash
uv sync
```

---

## Quick Start

### Run simulation experiments

```bash
python run_simulations.py
```

This script:

- Generates simulated datasets.
- Trains all methods.
- Estimates the Average Treatment Effect on the Treated (ATT).
- Saves results in `results/`.
- Generates evaluation plots in `figures/`.

### Run experiments on real-world data

```bash
python run_on_real_data.py
```

Before running, specify:

- dataset path,
- results path,
- model hyperparameters,
- number of bootstrap iterations.

---

## Using Your Own Dataset

Your dataset should contain the following columns:

```text
id
time
treatment
y
```

where:

- `id`: unit identifier
- `time`: time index
- `treatment`: binary treatment indicator (0/1)
- `y`: outcome variable

---
## Current Limitations

The current implementation focuses on the experimental settings presented in the paper and assumes a common treatment adoption time.

Support for staggered treatment adoption as shown in appendix is planned for a future release.

## Citation

If you use this repository, please cite:

```bibtex
@misc{maha2026balancedtwinscausalinference,
      title={Balanced Twins: Causal Inference on Time Series with Hidden Confounding}, 
      author={Ouali Maha and Ghattas Badih and Flachaire Emmanuel and Charpentier Philippe and Bozzi Laurent},
      year={2026},
      eprint={2606.18969},
      archivePrefix={arXiv},
      primaryClass={stat.ME},
      url={https://arxiv.org/abs/2606.18969}, 
}
```