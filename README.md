# ln2149-causal-bandits

Official implementation for the paper **Causal Contextual Bandits under Latent Confounding**.

## Citation

> Jennifer Baker, Tomas Lindqvist, Ana Duarte. "Causal Contextual Bandits under Latent Confounding." Accepted at AISTATS 2026.

## What is included

- `src/model.py` — model definition
- `src/train.py` — training entry point
- `src/rollout.py` — evaluation / rollout utilities
- `src/dataset.py`, `src/metrics.py`, `src/utils.py` — data and metric helpers
- `configs/` — experiment configuration
- `scripts/prepare_data.py`, `scripts/run_experiments.sh` — data prep and experiment driver
- `tests/` — smoke tests for the model and data pipeline

The repository contains the complete training and evaluation pipeline used for the
results reported in the paper. See `docs/setup.md` for environment setup and
`scripts/run_experiments.sh` for the full experiment driver.
