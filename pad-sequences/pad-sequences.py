import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    N = len(seqs)
    L = max_len if max_len is not None else (max(len(s) for s in seqs) if seqs else 0)

    # Tableau initialisé avec pad_value partout
    out = np.full((N, L), pad_value)

    # On copie chaque séquence dans sa ligne (tronquée si > L)
    for i, seq in enumerate(seqs):
        length = min(len(seq), L)
        out[i, :length] = seq[:length]

    return out  