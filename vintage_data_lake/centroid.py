import polars as pl
import numpy as np


def vectors_to_numpy(df, col="vector", dim=1024):
    return df[col].to_numpy()

def normalize_vector(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

def mean_vector(vectors: np.ndarray) -> np.ndarray:
    return vectors.mean(axis=0)


def calc_centroid(X: np.ndarray, trim_frac=0.0, max_iter = 2) -> np.ndarray:
    Xn = normalize_vectors(X)
    c = mean_vector(Xn)
    for _ in range(max_iter):
        if trim_frac > 0:
            cos = Xn @ c
            lo = np.quantile(cos, trim_frac)
            hi = np.quantile(cos, 1 - trim_frac)
            keep = (cos >= lo) & (cos <= hi)
            Xk = Xn[keep]
        else:
            Xk = Xn
        c = mean_vector(Xk)

    return c


def dispersion_and_exemplar(X, centroid, ids=None):
    Xn = normalize_vectors(X)
    cos = Xn @ centroid
    # dispersion: mean angular distance
    disp = (1.0 - cos).mean()
    # exemplar: nearest chunk to centroid
    idx = int(np.argmax(cos))
    exemplar_id = (ids[idx] if ids is not None else idx)
    return disp, exemplar_id

def bootstrap_centroid(X, w=None, B=200, trim_frac=0.0):
    N = X.shape[0]
    cents = []
    for _ in range(B):
        idx = np.random.randint(0, N, size=N)
        Xi = X[idx]
        cents.append(calc_centroid(Xi, trim_frac=trim_frac))
    return np.stack(cents, axis=0)  # [B, D]

def cosine(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a)*np.linalg.norm(b)) + 1e-12))