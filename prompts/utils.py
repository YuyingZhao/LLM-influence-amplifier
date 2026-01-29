"""Utility helpers for prompt generation and preprocessing."""

import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch


def pickle_load(file_name):
    """Load a pickle file.

    Args:
        file_name: Path to the pickle file.

    Returns:
        Deserialized object.
    """
    with open(file_name, 'rb') as f:
        return pickle.load(f)
    
def seed_everything(seed):
    """Seed RNGs for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
