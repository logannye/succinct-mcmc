# Succinct MCMC Documentation

Welcome to **Succinct MCMC**, a library for √T-space Markov chain trajectories.

## Quickstart
```python
from succinct_mcmc import SuccinctChain, StepFunctionKernel, split_rhat

kernel = StepFunctionKernel(lambda x, rng: x + rng.normalvariate(0.0, 1.0))
chain = SuccinctChain(kernel, initial_state=0.0, num_steps=100_000, warmup_steps=5_000)
chain.run()
print(chain.expectation(lambda x: x))
```

Use `succinct_mcmc.trace.iterator` for block-aware traversal and `succinct_mcmc.diagnostics` for R-hat/ESS calculations.

## Browse Topics
- [Design](design.md): Core ideas, height-compression, and architecture.
- [API Reference](api_reference.md): Public classes and functions.
- [Integration Guide](integration_guide.md): How to plug in your own samplers.
- [Examples](examples.md): End-to-end workflows.
