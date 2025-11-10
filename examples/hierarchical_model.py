"""
Example (sketch): Hierarchical model with succinct MCMC.

This file is a placeholder to illustrate:
- Using a structured state (e.g., dict or dataclass),
- Using GenericGibbsKernel or StepFunctionKernel over that state,
- Storing only succinct anchors while still supporting full diagnostics.

A full implementation would:
- Define conditional updaters for group-level and global parameters.
- Plug them into GenericGibbsKernel.
"""

from succinct_mcmc.mcmc import GenericGibbsKernel
from succinct_mcmc.trace import SuccinctChain
from succinct_mcmc.diagnostics import mean


def main():
    # Placeholder: toy state and updater.
    class ToyUpdater:
        def update(self, state, rng):
            # shift a "mu" field slightly as a fake conditional update
            new_mu = state["mu"] + rng.normalvariate(0.0, 0.1)
            return {**state, "mu": new_mu}

    initial_state = {"mu": 0.0}
    kernel = GenericGibbsKernel([ToyUpdater()])

    chain = SuccinctChain(kernel, initial_state, num_steps=50_000, master_seed=7)
    chain.run()

    mu_mean = mean(chain, lambda s: s["mu"])
    print("Toy hierarchical 'mu' mean:", mu_mean)


if __name__ == "__main__":
    main()
