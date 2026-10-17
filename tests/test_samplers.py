"""Tests for geogenie.samplers (sample weighting and synthetic oversampling)."""

import argparse
import inspect
import logging
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import torch

import geogenie.samplers.interpolate as I
import geogenie.samplers.samplers as S


class _StubPlot:
    def __init__(self, *args, **kwargs):
        pass


@pytest.fixture(autouse=True)
def _stub_plotting(monkeypatch, tmp_path):
    monkeypatch.setattr(I, "PlotGenIE", _StubPlot)
    # Any file written to the working directory ends up here.
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)


def _args(**kw):
    a = dict(
        output_dir=".",
        prefix="t",
        basemap_fips=None,
        highlight_basemap_counties=None,
        shapefile=None,
        show_plots=False,
        fontsize=12,
        filetype="png",
        plot_dpi=50,
        remove_splines=False,
        n_bins=3,
        oversample_method="kmeans",
        oversample_neighbors=5,
        use_weighted="none",
        verbose=0,
        seed=7,
    )
    a.update(kw)
    return argparse.Namespace(**a)


def _kw(cls, **kw):
    """Keep only the keyword arguments that ``cls`` accepts."""
    params = inspect.signature(cls.__init__).parameters
    return {k: v for k, v in kw.items() if k in params}


def _blobs(sizes, centers, spread=0.05, seed=0):
    rng = np.random.default_rng(seed)
    pts = [
        np.column_stack(
            [rng.normal(cx, spread, n), rng.normal(cy, spread, n)]
        )
        for n, (cx, cy) in zip(sizes, centers)
    ]
    return np.vstack(pts)


def _genotypes(n, n_loci=40, seed=0):
    return np.random.default_rng(seed).integers(0, 3, size=(n, n_loci)).astype(
        np.float32
    )


def _interp(y, X=None, w=None, **kw):
    X = _genotypes(len(y)) if X is None else X
    w = np.ones(len(y)) if w is None else w
    kw.setdefault("n_bins", 3)
    kw = _kw(I.GenotypeInterpolator, seed=kw.pop("seed", 7), **kw)
    return I.GenotypeInterpolator(X, y, w, _args(), **kw)


# Two large groups plus a small, isolated group of 2 samples.
Y_ISOLATED = _blobs([100, 100, 2], [(-92, 35), (-90, 35), (-86, 36)], spread=0.1)


def test_oversample_factor_is_capped():
    gi = _interp(Y_ISOLATED)
    Xo, yo, So = gi.interpolate_genotypes()
    n_syn = len(yo) - len(Y_ISOLATED)
    # Uncapped: the 2-sample cluster gets factor 50 -> 98 synthetic rows.
    assert n_syn <= 2 * (5 - 1)
    assert max(gi.oversample_factors.values()) == 5


def test_max_oversample_factor_none_is_uncapped():
    gi = _interp(Y_ISOLATED, max_oversample_factor=None)
    gi.interpolate_genotypes()
    assert max(gi.oversample_factors.values()) == 50


def test_neighbour_drawn_per_synthetic_child():
    # One small cluster (factor 5 -> 4 children per parent) with random coords.
    y = _blobs([200, 40], [(-92, 35), (-89, 35)], spread=0.2, seed=1)
    gi = _interp(y, n_bins=2)
    Xo, yo, So = gi.interpolate_genotypes()
    small = np.argmin(np.bincount(gi.original_cluster_labels))
    n_parents = np.sum(gi.original_cluster_labels == small)
    syn = np.asarray(gi.sample_origin_list) == "synthetic"
    n_unique = len(np.unique(np.round(yo[syn], 8), axis=0))
    # Old behaviour: one neighbour per parent -> at most n_parents coordinates.
    assert n_unique > 2 * n_parents


def test_self_is_not_a_neighbour():
    y = _blobs([60, 20], [(-92, 35), (-90, 35)], spread=0.2, seed=2)
    gi = _interp(y, n_bins=2)
    nb = gi.neighbor_indices
    assert not np.any(nb == np.arange(len(y))[:, None])
    Xo, yo, So = gi.interpolate_genotypes()
    syn = np.asarray(gi.sample_origin_list) == "synthetic"
    # A synthetic sample never sits exactly on a real sample (no selfing).
    d = np.min(np.abs(yo[syn][:, None, :] - y[None, :, :]).sum(-1), axis=1)
    assert np.all(d > 0)


def test_self_excluded_with_duplicate_coordinates():
    y = np.vstack([np.tile([[-92.0, 35.0]], (8, 1)), _blobs([20], [(-91, 35)])])
    gi = _interp(y, n_bins=2)
    assert gi.neighbor_indices.shape == (len(y), gi.n_neighbors)
    assert not np.any(gi.neighbor_indices == np.arange(len(y))[:, None])


def test_hybrid_continuous_features_are_midpoints():
    gi = _interp(Y_ISOLATED)
    # First four positions are continuous (e.g., PCA); the rest are 0/1/2.
    g1 = np.array([-0.43, -0.09, 0.59, 1.49] + [0, 0, 0, 1, 2, 2] * 50, dtype=np.float32)
    g2 = np.array([0.40, 2.93, -0.62, -1.12] + [0, 2, 1, 1, 2, 1] * 50, dtype=np.float32)
    h = gi._vectorized_sample_hybrid(g1, g2)
    np.testing.assert_allclose(h[:4], (g1[:4] + g2[:4]) / 2, rtol=1e-6)
    h, g1, g2 = h[4:], g1[4:], g2[4:]
    assert set(np.unique(h)) <= {0.0, 1.0, 2.0}
    hom_same = (g1 == g2) & (g1 != 1)
    assert np.all(h[hom_same] == g1[hom_same])
    assert np.all(h[(g1 == 0) & (g2 == 2)] == 1)
    assert np.all(np.isin(h[(g1 == 0) & (g2 == 1)], [0, 1]))


def test_no_hets_strategy():
    gi = _interp(Y_ISOLATED)
    g1 = np.array([0, 0, 1, 2] * 25, dtype=np.float32)
    g2 = np.array([2, 1, 1, 2] * 25, dtype=np.float32)
    h = gi._vectorized_sample_hybrid(g1, g2, sampling_strategy="no_hets")
    differ = g1 != g2
    assert set(np.unique(h[differ])) <= {0.0, 2.0}
    assert np.all(h[~differ] == g1[~differ])


def test_shuffle_permutes_rows_and_keeps_metadata_aligned():
    y = _blobs([60, 20], [(-92, 35), (-90, 35)], spread=0.2, seed=3)
    X = _genotypes(len(y), seed=3)
    gi = _interp(y, X=X, n_bins=2)
    Xo, yo, So = gi.interpolate_genotypes()
    origin = np.asarray(gi.sample_origin_list)
    assert len(origin) == len(yo) == len(gi.cluster_labels)
    orig = origin == "original"
    assert orig.sum() == len(y)
    # Rows are shuffled (the original rows are not the leading block in order).
    assert not np.all(np.diff(gi.parent_indices) >= 0)
    # Original rows carry their own genotype and coordinates.
    np.testing.assert_array_equal(Xo[orig], X[gi.parent_indices[orig]])
    np.testing.assert_array_equal(yo[orig], y[gi.parent_indices[orig]])
    # Synthetic rows sit at the midpoint of their two parents.
    syn = ~orig
    mid = (y[gi.parent_indices[syn]] + y[gi.mate_indices[syn]]) / 2
    np.testing.assert_allclose(yo[syn], mid)
    # Cluster labels follow the rows.
    np.testing.assert_array_equal(
        gi.cluster_labels[orig], gi.original_cluster_labels[gi.parent_indices[orig]]
    )


def test_interpolator_is_reproducible_with_seed():
    y = _blobs([60, 20], [(-92, 35), (-90, 35)], spread=0.2, seed=4)
    out = []
    for g in (0, 123):
        np.random.seed(g)  # global RNG state must not matter
        gi = _interp(y, n_bins=2, seed=11)
        out.append(gi.interpolate_genotypes())
    for a, b in zip(*out):
        np.testing.assert_array_equal(a, b)


def test_int_n_neighbors():
    gi = _interp(Y_ISOLATED, n_neighbors=3)
    assert gi.n_neighbors == 3
    assert gi.neighbor_indices.shape == (len(Y_ISOLATED), 3)


def test_embedded_flag_does_not_crash():
    gi = _interp(Y_ISOLATED, embedded=True)
    Xo, yo, So = gi.interpolate_genotypes()
    assert len(yo) > len(Y_ISOLATED)


def test_nbins_none_silhouette_uses_latlon_radians(monkeypatch):
    seen = []
    real = I.silhouette_score

    def spy(X, labels, **kw):
        seen.append(np.asarray(X).copy())
        return real(X, labels, **kw)

    monkeypatch.setattr(I, "silhouette_score", spy)
    y = _blobs([30, 30, 30], [(-92, 35), (-90, 36), (-91, 34)], spread=0.1)
    _interp(y, n_bins=None)
    assert seen
    np.testing.assert_allclose(seen[0], np.radians(y[:, ::-1]))


def test_kerneldensity_branch_uses_latlon_radians_and_oversamples_sparse():
    y = np.vstack(
        [
            _blobs([80], [(-92, 35)], spread=0.05, seed=5),
            _blobs([20], [(-90, 35)], spread=0.8, seed=6),
        ]
    )
    gi = _interp(y, use_kde_densities=True)
    fitted = np.asarray(gi.kde.tree_.data)
    np.testing.assert_allclose(fitted, np.radians(y[:, ::-1]))
    labels = gi.original_cluster_labels
    # Label 0 (below the density threshold) is the minority, so it is oversampled.
    assert np.sum(labels == 0) < np.sum(labels == 1)
    gi.interpolate_genotypes()
    assert gi.oversample_factors[0] >= gi.oversample_factors[1] == 1


# ---------------------------------------------------------------- weights


def _gds(y, **kw):
    kw = _kw(S.GeographicDensitySampler, **kw)
    return S.GeographicDensitySampler(pd.DataFrame(y, columns=["x", "y"]), **kw)


Y5 = _blobs(
    [60, 60, 60, 60, 60],
    [(-94, 33), (-90, 33), (-94, 36), (-90, 36), (-92, 34.5)],
    spread=0.1,
)


def test_kmeans_elbow_finds_k():
    np.random.seed(0)
    g = _gds(Y5, use_kmeans=True, use_kde=False, max_clusters=10, seed=0)
    assert g.find_optimal_clusters() == 5
    g.max_clusters = 4
    assert g.find_optimal_clusters() <= 4


def test_weights_mean_one_without_normalize():
    np.random.seed(0)
    g = _gds(Y5[:250], use_kmeans=True, use_kde=False, normalize=False, seed=0)
    assert np.isclose(g.weights.mean(), 1.0)
    g = _gds(Y5[:250], use_kmeans=False, use_kde=True, normalize=False, seed=0)
    assert np.isclose(g.weights.mean(), 1.0)


def test_warns_when_no_weighting_method(caplog):
    with caplog.at_level(logging.WARNING):
        g = _gds(Y5, use_kmeans=False, use_kde=False, use_dbscan=False)
    assert np.all(g.weights == 1)
    assert any("all sample weights are 1" in r.getMessage() for r in caplog.records)


def test_kde_bandwidth_parameter():
    g_default = _gds(Y5, use_kmeans=False, use_kde=True)
    assert np.isclose(g_default.kde_bandwidth, np.radians(0.03))
    g_wide = _gds(Y5, use_kmeans=False, use_kde=True, kde_bandwidth=np.radians(1.0))
    assert np.isclose(g_wide.kde_bandwidth, np.radians(1.0))
    assert not np.allclose(g_default.density, g_wide.density)


def test_density_sampler_reproducible_with_seed():
    # Uniform points: the KMeans solution depends on the initialisation.
    y = np.random.default_rng(5).uniform([-94, 33], [-90, 36], size=(300, 2))
    w = []
    for g in (0, 99):
        np.random.seed(g)  # global RNG state must not matter
        w.append(_gds(y, use_kmeans=True, use_kde=False, seed=3).weights)
    np.testing.assert_array_equal(w[0], w[1])


def test_weight_plot_not_written_to_cwd(tmp_path):
    out = tmp_path / "out"
    _gds(Y5, use_kmeans=False, use_kde=True, output_dir=str(out), prefix="run1")
    _gds(Y5, use_kmeans=False, use_kde=True)
    assert not os.path.exists("sample_weights.png")
    assert (out / "plots" / "run1_sample_weights.png").exists()


def test_sampler_iter_and_len_use_indices():
    g = _gds(Y5, use_kmeans=False, use_kde=True, seed=0)
    g.indices = np.arange(10, 20)
    g.weights = np.ones(10)
    drawn = list(iter(g))
    assert len(g) == 10
    assert len(drawn) == 10
    assert set(drawn) <= set(range(10, 20))


# ------------------------------------------------------- resample_interp


class _Norm:
    def transform(self, y):
        return np.asarray(y)

    def inverse_transform(self, y):
        return np.asarray(y)


class _DS:
    norm = _Norm()


def _loader(y, w):
    ds = argparse.Namespace(
        features=torch.tensor(_genotypes(len(y))),
        labels=torch.tensor(y, dtype=torch.float32),
        sample_weights=torch.tensor(w, dtype=torch.float32),
    )
    return argparse.Namespace(dataset=ds, batch_size=16)


def test_resample_interp_wires_neighbours_and_recomputes_weights():
    y = _blobs([80, 20], [(-92, 35), (-90, 35)], spread=0.2, seed=8)
    w = _gds(y, use_kmeans=False, use_kde=True, normalize=True).weights
    args = _args(
        n_bins=2,
        oversample_neighbors=3,
        use_weighted="loss",
        use_kmeans=False,
        use_kde=True,
        use_dbscan=False,
        w_power=1.0,
        max_clusters=10,
        max_neighbors=50,
        normalize_sample_weights=True,
        focus_regions=None,
    )
    centroids, gi, features, labels, sample_weights, indices = I.resample_interp(
        _loader(y, w), args, _DS()
    )
    assert gi.n_neighbors == 3
    expected = _gds(
        np.asarray(labels, dtype=float), use_kmeans=False, use_kde=True, normalize=True
    ).weights
    np.testing.assert_allclose(sample_weights, expected, rtol=1e-5)
