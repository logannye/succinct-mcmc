from rewind.core.blocks import Block
from rewind.core.seeds import derive_seed
from rewind.core.replay import replay_to, iter_block


def _step(x, rng):
    return x + (1 if rng.random() < 0.5 else -1)


def test_replay_to_start_returns_anchor():
    block = Block(0, 0, 10)
    assert replay_to(_step, anchor_state=0, block=block, target=0,
                     seed=derive_seed(1, 0)) == 0


def test_replay_to_matches_iter_block():
    block = Block(1, 10, 20)
    seed = derive_seed(7, 1)
    states = list(iter_block(_step, anchor_state=99, block=block, seed=seed))
    assert len(states) == 10            # X_10 .. X_19
    assert states[0] == 99
    for offset, target in enumerate(range(block.start, block.stop)):
        assert replay_to(_step, anchor_state=99, block=block, target=target, seed=seed) == states[offset]
