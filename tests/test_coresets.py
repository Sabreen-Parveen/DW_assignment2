"""Sanity checks for the coreset constructions. Run with: python -m pytest tests"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from coresets import kmeans_cost, lightweight_coreset, sensitivity_coreset, uniform_sample  # noqa: E402


def make_data(seed=0):
    rng = np.random.default_rng(seed)
    big = rng.normal(0, 1, size=(5000, 3))
    small = rng.normal(20, 1, size=(50, 3))  # a small, far-away cluster
    return np.vstack([big, small])


def test_weights_sum_to_n_on_average():
    X = make_data()
    sums = [lightweight_coreset(X, 500, np.random.default_rng(s))[1].sum() for s in range(200)]
    assert abs(np.mean(sums) / len(X) - 1) < 0.02


def test_weighted_cost_is_unbiased():
    X = make_data()
    centers = np.array([[0, 0, 0], [20, 20, 20]], dtype=float)
    true = kmeans_cost(X, centers)
    est = []
    for s in range(200):
        C, w = lightweight_coreset(X, 500, np.random.default_rng(s))
        d2 = ((C[:, None, :] - centers[None]) ** 2).sum(-1).min(1)
        est.append((w * d2).sum())
    assert abs(np.mean(est) / true - 1) < 0.03


def test_lightweight_samples_far_cluster_more_than_uniform():
    X = make_data()
    hits_lw, hits_un = 0, 0
    for s in range(100):
        rng = np.random.default_rng(s)
        C, _ = lightweight_coreset(X, 200, rng)
        hits_lw += (C[:, 0] > 10).sum()
        C, _ = uniform_sample(X, 200, np.random.default_rng(s))
        hits_un += (C[:, 0] > 10).sum()
    assert hits_lw > 5 * hits_un


def test_sensitivity_coreset_shapes():
    X = make_data()
    C, w = sensitivity_coreset(X, 300, k=5, rng=np.random.default_rng(0))
    assert C.shape == (300, 3) and w.shape == (300,) and (w > 0).all()
