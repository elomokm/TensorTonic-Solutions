import numpy as np

def euclidean_distance(x: list, y: list) -> float:
    """
    Returns the Euclidean distance as a float.
    """
    v1 = np.asarray(x)
    v2 = np.asarray(y)
    norm = (v1-v2)**2
    dist = np.sqrt(sum(norm))
    return float(dist)