import rewind


def walk(x, rng):
    return x + (1 if rng.random() < 0.5 else -1)


def test_branch_records_parent_hash():
    run = rewind.record(walk, init_state=0, n_steps=100, seed=4)
    parent_hash = run.content_hash()
    fork = run.branch(40)
    assert fork.parent_hash == parent_hash
    assert fork.branch_point == 40


def test_content_hash_is_stable():
    run = rewind.record(walk, init_state=0, n_steps=50, seed=4)
    assert run.content_hash() == run.content_hash()


def test_grandchild_chains_back():
    run = rewind.record(walk, init_state=0, n_steps=120, seed=4)
    child = run.branch(40)
    grandchild = child.branch(20)
    assert grandchild.parent_hash == child.content_hash()
    assert grandchild.parent_hash != run.content_hash()
