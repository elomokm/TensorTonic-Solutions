import numpy as np

def vector_norms(v: list) -> np.ndarray:
    """
    Returns a float64 array containing the L1, L2, and infinity norms.
    """
    x = np.asarray(v, dtype=np.float64)
    a = np.abs(x)
    l1 = a.sum()
    l2 = np.sqrt((x * x).sum())
    linf = a.max()
    return np.array([l1, l2, linf])