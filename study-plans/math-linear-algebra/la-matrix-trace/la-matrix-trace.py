import numpy as np

def matrix_trace(A: list) -> float:
    """
    Returns the trace as a Python float.
    """
    A = np.asarray(A , dtype=np.float64)
    res = 0
    for i in range(A.shape[0]):
        for j in range (A.shape[1]):
            if i == j:
                res+=A[i][j]
    return float(res) 