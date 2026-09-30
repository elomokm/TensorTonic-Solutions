import numpy as np

def hadamard_product(A: list, B: list) -> np.ndarray:
    """
    Returns the element-wise product as a float64 array.
    """
    v1 = np.asarray(A, dtype=np.float64)
    v2 = np.asarray(B, dtype=np.float64)
    return v1 * v2