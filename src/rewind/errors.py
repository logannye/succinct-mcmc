"""rewind error types."""


class RewindError(Exception):
    """Base class for rewind errors."""


class NondeterministicStepError(RewindError):
    """A step_fn did not reproduce a stored anchor during recording (self_check)."""

    def __init__(self, block_index: int):
        super().__init__(
            f"step_fn is non-deterministic: block {block_index} did not reproduce "
            f"its stored next anchor. Ensure all randomness flows through the rng "
            f"and there is no hidden/global/external state."
        )
        self.block_index = block_index


class NondeterministicReplayError(RewindError):
    """Replay from an artifact did not reproduce a stored anchor on this machine."""

    def __init__(self, block_index: int):
        super().__init__(
            f"replay mismatch at block {block_index}: this machine/code does not "
            f"reproduce the recorded run bit-for-bit."
        )
        self.block_index = block_index


class ArtifactIntegrityError(RewindError):
    """An artifact failed its content-hash or schema check on load."""
