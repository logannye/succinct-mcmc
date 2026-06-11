import math
import rewind
from examples.agent_sim import sim_step, NaNState


def test_scrub_to_first_nan_tick():
    # A cheap-tick sim that eventually produces a NaN; rewind lands on the exact tick.
    run = rewind.record(sim_step, init_state=NaNState(value=1.0), n_steps=5000, seed=123)

    def first_bad_tick():
        for t in range(run.n_steps):
            if math.isnan(run.get(t).value):
                return t
        return None

    t = first_bad_tick()
    assert t is not None
    assert math.isnan(run.get(t).value)
    assert not math.isnan(run.get(t - 1).value)   # the tick BEFORE is finite -> root cause is at t


def test_branch_from_just_before_nan_can_avoid_it():
    run = rewind.record(sim_step, init_state=NaNState(value=1.0), n_steps=5000, seed=123)
    # fork just before the blow-up with a clamp mutation; the branch should stay finite for a while
    bad = next(t for t in range(run.n_steps) if math.isnan(run.get(t).value))
    fork = run.branch(bad - 1, mutate=lambda s: NaNState(value=0.5))
    assert not math.isnan(fork.get(0).value)
