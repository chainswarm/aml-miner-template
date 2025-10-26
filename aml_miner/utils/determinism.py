import random
import os
import numpy as np
from loguru import logger


def set_deterministic_mode(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    logger.info(f"Deterministic mode enabled with seed={seed}")