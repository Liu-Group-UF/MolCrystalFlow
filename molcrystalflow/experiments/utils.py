"""Utility functions for experiments."""

import logging

import GPUtil
from pytorch_lightning.utilities.rank_zero import rank_zero_only


def get_available_device(num_device: int) -> list:
    """Get available GPU devices sorted by memory.

    Args:
        num_device: Number of devices to return.

    Returns:
        List of device IDs.
    """
    return GPUtil.getAvailable(order='memory', limit=8)[:num_device]


def get_pylogger(name: str = __name__) -> logging.Logger:
    """Initialize a multi-GPU-friendly python command line logger.

    Args:
        name: Logger name.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Mark all logging levels with rank zero decorator to prevent
    # duplicate logs in multi-GPU setup
    logging_levels = (
        "debug", "info", "warning", "error",
        "exception", "fatal", "critical"
    )
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def flatten_dict(raw_dict: dict) -> list:
    """Flatten a nested dictionary.

    Args:
        raw_dict: Nested dictionary to flatten.

    Returns:
        List of (key, value) tuples with flattened keys.
    """
    flattened = []
    for k, v in raw_dict.items():
        if isinstance(v, dict):
            flattened.extend([
                (f'{k}:{i}', j) for i, j in flatten_dict(v)
            ])
        else:
            flattened.append((k, v))
    return flattened
