"""
Coreset constructions for k-means.

- lightweight_coreset: Algorithm 1 of Bachem, Lucic & Krause (KDD 2018),
  "Scalable k-Means Clustering via Lightweight Coresets", arXiv:1702.08248.
- sensitivity_coreset: the "CS" baseline of the paper (Lucic et al., 2016):
  k-means++ seeding followed by importance sampling on sensitivity bounds.
- uniform_sample: the "Uniform" baseline.

Every function returns (C, w): the m sampled points and their weights.
"""

import numpy as np


def lightweight_coreset(X, m, rng):
    """Algorithm 1: sample with q(x) = 1/2 * 1/n + 1/2 * d(x, mu)^2 / sum d(x', mu)^2."""
    n = X.shape[0]
    mu = X.mean(axis=0)
    dist2 = ((X - mu) ** 2).sum(axis=1)
    q = 0.5 / n + 0.5 * dist2 / dist2.sum()
    idx = rng.choice(n, size=m, replace=True, p=q)
    return X[idx], 1.0 / (m * q[idx])


def uniform_sample(X, m, rng):
    """Uniform subsample; each point stands for n/m points."""
    n = X.shape[0]
    idx = rng.choice(n, size=m, replace=True)
    return X[idx], np.full(m, n / m)


def kmeans_pp_seeding(X, k, rng):
    """D^2 sampling (k-means++ seeding). Returns centre indices and each point's
    nearest-centre index and squared distance."""
    n = X.shape[0]
    centers = [rng.integers(n)]
    dist2 = ((X - X[centers[0]]) ** 2).sum(axis=1)
    assign = np.zeros(n, dtype=np.int64)
    for j in range(1, k):
        c = rng.choice(n, p=dist2 / dist2.sum())
        centers.append(c)
        d_new = ((X - X[c]) ** 2).sum(axis=1)
        closer = d_new < dist2
        dist2[closer] = d_new[closer]
        assign[closer] = j
    return np.array(centers), assign, dist2


def sensitivity_coreset(X, m, k, rng):
    """CS baseline: bound each point's sensitivity using a k-means++ solution B,
    then sample proportionally (Lucic et al. 2016; Bachem et al. 2017, Alg. 2)."""
    n = X.shape[0]
    _, assign, dist2 = kmeans_pp_seeding(X, k, rng)
    alpha = 16 * (np.log(k) + 2)
    c_phi = dist2.mean()
    cluster_size = np.bincount(assign, minlength=k)
    cluster_cost = np.bincount(assign, weights=dist2, minlength=k)
    s = (alpha * dist2 / c_phi
         + 2 * alpha * cluster_cost[assign] / (cluster_size[assign] * c_phi)
         + 4 * n / cluster_size[assign])
    q = s / s.sum()
    idx = rng.choice(n, size=m, replace=True, p=q)
    return X[idx], 1.0 / (m * q[idx])


def kmeans_cost(X, centers, batch=20000):
    """Quantization error phi_X(Q): sum of squared distances to the nearest centre."""
    total = 0.0
    c2 = (centers ** 2).sum(axis=1)
    for i in range(0, X.shape[0], batch):
        xb = X[i:i + batch]
        d2 = (xb ** 2).sum(axis=1)[:, None] - 2 * xb @ centers.T + c2[None, :]
        total += np.maximum(d2.min(axis=1), 0).sum()
    return total
