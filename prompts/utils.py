import pickle
import random
import numpy as np
import torch
import os
from collections import defaultdict


def pickle_load(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)
    
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    