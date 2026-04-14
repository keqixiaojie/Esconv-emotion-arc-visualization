import numpy as np
from typing import Optional

def smooth_scores(scores: np.ndarray, window_size: int, method: str = 'mean') -> Optional[np.ndarray]:
    """
    对离散情感分数进行滑动窗口平滑
    
    Args:
        scores: 离散分数数组
        window_size: 窗口大小
        method: 平滑方法 ('mean' 或 'gaussian')
        
    Returns:
        平滑后的弧线数组，若分数不足则返回 None
    """
    if len(scores) < window_size:
        return None
    
    if method == 'mean':
        kernel = np.ones(window_size) / window_size
        return np.convolve(scores, kernel, mode='valid')
    
    elif method == 'gaussian':
        from scipy.ndimage import gaussian_filter1d
        sigma = window_size / 4.0
        return gaussian_filter1d(scores, sigma=sigma)
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")