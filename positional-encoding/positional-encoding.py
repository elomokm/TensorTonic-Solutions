import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    pe = np.zeros((seq_len, d_model))
    positions = np.arange(seq_len)[:, np.newaxis]

    # sin uses ceil(d_model/2) pairs, cos uses floor(d_model/2)
    i_sin = np.arange((d_model + 1) // 2)[np.newaxis, :]
    i_cos = np.arange(d_model // 2)[np.newaxis, :]

    pe[:, 0::2] = np.sin(positions / (base ** (2 * i_sin / d_model)))
    pe[:, 1::2] = np.cos(positions / (base ** (2 * i_cos / d_model)))

    return pe