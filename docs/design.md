# Design Overview

Succinct MCMC is built on three core principles:

1. **Markov structure**  
   Every MCMC step is `X_{t+1} = F(X_t, U_t)` with fixed-size state.

2. **Deterministic replay**  
   Given:
   - an anchor state,
   - a PRNG seed,
   we can deterministically regenerate all steps within a block.

3. **Height-compressed layout**  
   For a chain of length `T`, we:
   - choose block size `b ≈ √T`,
   - store `O(T / b) = O(√T)` anchors,
   - re-derive randomness per block from a master seed.

This realizes a **time–space tradeoff**:
- Memory: `O(√T · d)` instead of `O(T · d)`
- Extra compute: replays within blocks when accessing samples or computing summaries.

See the source in `src/succinct_mcmc/core/` and `src/succinct_mcmc/trace/` for the implementation skeleton.
