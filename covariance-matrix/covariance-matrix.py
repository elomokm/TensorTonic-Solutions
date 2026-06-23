import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    X = np.asarray(X, dtype=float)

    # Entrée invalide : pas 2D
    if X.ndim != 2:
        return None

    N, D = X.shape

    # Entrée invalide : moins de 2 échantillons
    if N < 2:
        return None

    # Étape 1 : centrer les données (moyenne par colonne)
    mu = np.mean(X, axis=0)        # shape (D,)
    X_c = X - mu                   # broadcasting -> shape (N, D)

    # Étape 2 : matrice de covariance (covariance d'échantillon, division par N-1)
    S = (X_c.T @ X_c) / (N - 1)    # shape (D, D)

    return S