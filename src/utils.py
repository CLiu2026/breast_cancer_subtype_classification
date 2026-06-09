import numpy as np
import random
import os

def set_seed(seed=42):
    """固定随机种子，保证可复现"""
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def ensure_dir(path):
    """若目录不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path)