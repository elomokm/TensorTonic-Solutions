import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a float.
    """
    a = np.asarray (a , dtype=np.float64)
    b = np.asarray (b , dtype=np.float64)
    normA = float(np.linalg.norm(a,2))
    normB = float(np.linalg.norm(b,2))
    product = normA*normB
    if product != 0 :
        cos_sim = float((a@b)/(normA*normB))
    else:
        cos_sim= 0
    
    return float(cos_sim)