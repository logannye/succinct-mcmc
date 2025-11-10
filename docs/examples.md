# Examples

Succinct MCMC ships with a handful of self-contained scripts under `examples/`:

- `gaussian_mean.py` — Run a long random-walk chain on a standard normal target, compute posterior mean, and inspect anchored samples.
- `logistic_regression.py` — Demonstrates succinct storage for Bayesian logistic regression using a wrapped step function kernel.
- `hierarchical_model.py` — Illustrates a Gibbs-style update on a toy hierarchical state.
- `hmc_bigdim.py` — Sketch for high-dimensional HMC (scalar kernel included; extend with vector backends).
- `multi_chain_diagnostics.ipynb` — Notebook showing multi-chain setup, R-hat/ESS diagnostics, and artifact export.

Each script accepts optional arguments for number of steps and seeds; inspect the source for details. Run them with:
```bash
python examples/gaussian_mean.py --help
```
