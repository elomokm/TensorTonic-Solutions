import numpy as np

def dot_product(x: list, y: list) -> float:
    """
    Returns the dot product as a float.
    """
    v = np.asarray(x)
    w = np.asarray(y)
    return float(v@w)