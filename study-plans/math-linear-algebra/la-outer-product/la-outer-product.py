import numpy as np

def outer_product(u: list, v: list) -> np.ndarray:
    """
    Returns the float64 outer-product matrix.
    """
    u = np.asarray(u , dtype = np.float64)
    v = np.asarray(v , dtype = np.float64)
    return np.outer(u,v)