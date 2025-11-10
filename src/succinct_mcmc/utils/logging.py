"""
Lightweight logging helpers.

We want:
- Minimal dependency footprint.
- Consistent debug/info output for long runs.

Later:
- Hook into standard logging,
- Allow users to set verbosity.
"""

import logging


def get_logger(name: str = "succinct_mcmc") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
