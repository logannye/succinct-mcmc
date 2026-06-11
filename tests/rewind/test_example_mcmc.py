import rewind
from examples.mcmc import metropolis_step


def test_metropolis_samples_standard_normal_mean_near_zero():
    run = rewind.record(metropolis_step, init_state=0.0, n_steps=20_000, seed=1)
    s = run.stats(lambda x: x)
    assert abs(s["mean"]) < 0.1
    assert run.get(12345) == list(run.iter())[12345]   # exact replay holds for MCMC too


def test_metropolis_is_pure_under_self_check():
    rewind.record(metropolis_step, init_state=0.0, n_steps=2_000, seed=1, self_check=True)
