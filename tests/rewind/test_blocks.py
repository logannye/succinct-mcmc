from rewind.core.blocks import Block, default_block_size, make_blocks, block_index_for_step


def test_default_block_size_is_ceil_sqrt():
    assert default_block_size(100) == 10
    assert default_block_size(101) == 11   # ceil(sqrt)
    assert default_block_size(1) == 1


def test_make_blocks_partitions_contiguously():
    blocks = make_blocks(n_steps=25, block_size=10)
    assert [(b.index, b.start, b.stop) for b in blocks] == [
        (0, 0, 10), (1, 10, 20), (2, 20, 25),
    ]


def test_make_blocks_exact_multiple():
    blocks = make_blocks(n_steps=20, block_size=10)
    assert [(b.start, b.stop) for b in blocks] == [(0, 10), (10, 20)]


def test_block_index_for_step():
    assert block_index_for_step(0, 10) == 0
    assert block_index_for_step(9, 10) == 0
    assert block_index_for_step(10, 10) == 1
    assert block_index_for_step(24, 10) == 2
