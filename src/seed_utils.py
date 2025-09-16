# src/seed_utils.py

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True):
    """
    Фиксирует сиды для всех библиотек, использующих генераторы случайных чисел.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
