"""Tests for the target scaler, data splitting, sample weights, allele filtering, sample-file parsing,
bootstrap resampling and Optuna search-space handling."""

import logging
import types

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from geogenie.utils.data import CustomDataset
from geogenie.utils.data_structure import DataStructure
from geogenie.utils.transformers import MinMaxScalerGeo
from geogenie.utils.utils import read_csv_with_dynamic_sep


def haversine_km(a, b):
    lon1, lat1, lon2, lat2 = map(np.radians, (a[:, 0], a[:, 1], b[:, 0], b[:, 1]))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(h))


@pytest.fixture
def coords():
    rng = np.random.default_rng(0)
    return np.c_[rng.uniform(-94.6, -89.9, 300), rng.uniform(33.0, 36.5, 300)]


# --- target scaler -----------------------------------------------------------------------------


def test_scaler_uses_training_range(coords):
    z = MinMaxScalerGeo().fit_transform(coords)
    assert z.min() >= -1e-12 and z.max() <= 1 + 1e-12
    assert z[:, 0].max() - z[:, 0].min() > 0.5  # spread over the scaled range, not a sliver
    assert np.allclose(MinMaxScalerGeo().fit(coords).inverse_transform(z), coords)


def test_scaler_is_distance_preserving(coords):
    z = MinMaxScalerGeo().fit_transform(coords)
    i, j = np.arange(0, 150), np.arange(150, 300)
    ratio = np.linalg.norm(z[i] - z[j], axis=1) / haversine_km(coords[i], coords[j])
    assert ratio.std() / ratio.mean() < 0.02


def test_scaler_constant_axis(coords):
    c = coords.copy()
    c[:, 1] = 35.0
    m = MinMaxScalerGeo().fit(c)
    assert np.allclose(m.inverse_transform(m.transform(c)), c)


# --- splitting ---------------------------------------------------------------------------------


def make_ds(coords, seed=1):
    ds = DataStructure.__new__(DataStructure)
    ds.y = coords
    ds.logger = logging.getLogger("test")
    ds.verbose = 0
    return ds


@pytest.mark.parametrize("n", [60, 150, 1400])
def test_cluster_split_exact_proportions(n):
    rng = np.random.default_rng(n)
    c = np.c_[rng.uniform(-94, -90, n), rng.uniform(33, 36.5, n)]
    ds = make_ds(c)
    args = types.SimpleNamespace(maxk=8, seed=3)
    first, second = ds._cluster_split_positions(c, 0.8, 3, args)
    assert len(first) == round(0.8 * n)
    assert len(set(first) & set(second)) == 0
    assert sorted(np.r_[first, second]) == list(range(n))


def test_cluster_split_reproducible(coords):
    ds = make_ds(coords)
    args = types.SimpleNamespace(maxk=6, seed=5)
    a = ds._cluster_split_positions(coords, 0.7, 5, args)
    b = ds._cluster_split_positions(coords, 0.7, 5, args)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_optimal_clusters_returns_best_k():
    rng = np.random.default_rng(0)
    centres = np.array([[0, 0], [10, 0], [0, 10], [10, 10], [20, 0], [20, 10]], float)
    pts = np.vstack([c + rng.normal(0, 0.3, (40, 2)) for c in centres])
    ds = make_ds(pts)
    assert ds._find_optimal_clusters(pts, types.SimpleNamespace(maxk=10, seed=0)) == 6


# --- minor allele count ------------------------------------------------------------------------


def test_min_mac_uses_minor_allele():
    ds = DataStructure.__new__(DataStructure)
    ds.logger = logging.getLogger("test")
    ds.verbose = 0
    # genotype-class counts per locus: [hom ref, het, hom alt]
    counts = np.array([[0, 0, 10], [0, 1, 9], [9, 1, 0], [5, 3, 2]])
    gt = np.arange(4)[:, None] * np.ones((4, 10))
    kept = ds.filter_gt(gt, min_mac=2, max_snps=None, allele_counts=counts)
    assert kept[:, 0].tolist() == [3.0]


# --- sample file parsing -----------------------------------------------------------------------


def test_sample_ids_read_as_text(tmp_path):
    f = tmp_path / "coords.csv"
    f.write_text("sampleID,x,y\n0012,-92.1,35.0\n13 ,-91.0,nan\n")
    df = read_csv_with_dynamic_sep(str(f))
    assert df.sampleID.tolist() == ["0012", "13"]
    assert df.x.dtype.kind == "f" and np.isnan(df.y.iloc[1])


# --- bootstrap resampling ----------------------------------------------------------------------


def make_bootstrap(method, seed=7):
    from geogenie.optimize.bootstrap import Bootstrap

    b = Bootstrap.__new__(Bootstrap)
    b.args = types.SimpleNamespace(bootstrap_method=method, use_weighted="loss", batch_size=8)
    b.dtype = torch.float32
    b.ds = types.SimpleNamespace(data={"pred_samples": np.array(["u1", "u2"])})
    b.logger = logging.getLogger("test")
    return b


def loaders():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 3, (40, 12)).astype(float)
    y = rng.normal(size=(40, 2))
    tr = DataLoader(CustomDataset(X, y, sample_weights=np.arange(40, dtype=float)), batch_size=8)
    va = DataLoader(CustomDataset(X[:10], y[:10], sample_ids=np.arange(10).astype(str)), batch_size=8)
    return tr, va, va, X[:2]


@pytest.mark.parametrize("method", ["samples", "loci", "none"])
def test_bootstrap_methods(method):
    tr, va, te, Xp = loaders()
    out = make_bootstrap(method)._resample_loaders(tr, va, te, Xp, replicate_seed=11)
    ftr = out[0].dataset.features.numpy()
    if method == "loci":
        assert ftr.shape == (40, 12)
        assert len({tuple(c) for c in ftr.T}) < 12  # some loci drawn more than once
    elif method == "samples":
        assert ftr.shape == (40, 12)
        assert len({tuple(r) for r in ftr}) < 40  # some samples drawn more than once
        w = out[0].dataset.sample_weights.numpy()
        orig = {tuple(r): wi for r, wi in zip(tr.dataset.features.numpy(), tr.dataset.sample_weights.numpy())}
        assert all(orig[tuple(r)] == wi for r, wi in zip(ftr, w))  # weights follow their samples
    else:
        assert np.array_equal(ftr, tr.dataset.features.numpy())
    # loci used for val/test/pred match training
    assert out[1].dataset.features.shape[1] == ftr.shape[1] == out[3].dataset.features.shape[1]


@pytest.mark.parametrize("method", ["samples", "loci"])
def test_bootstrap_resampling_reproducible(method):
    tr, va, te, Xp = loaders()
    a = make_bootstrap(method)._resample_loaders(tr, va, te, Xp, replicate_seed=11)[0].dataset.features
    b = make_bootstrap(method)._resample_loaders(tr, va, te, Xp, replicate_seed=11)[0].dataset.features
    assert torch.equal(a, b)


def test_reset_weights_changes_parameters():
    from geogenie.models.models import MLPRegressor
    from geogenie.optimize.bootstrap import Bootstrap

    m = MLPRegressor(input_size=5, width=8, nlayers=4, dropout_prop=0.1, device="cpu", output_width=2, dtype=torch.float32)
    before = [p.detach().clone() for p in m.parameters() if p.dim() > 1]
    Bootstrap.reset_weights(None, m)
    after = [p.detach() for p in m.parameters() if p.dim() > 1]
    assert any(not torch.equal(a, b) for a, b in zip(before, after))


# --- Optuna search space -----------------------------------------------------------------------


@pytest.mark.parametrize("n_features", [5, 9, 419, 50000])
def test_width_range_valid(n_features):
    import optuna

    from geogenie.optimize.optuna_opt import Optimize

    o = Optimize.__new__(Optimize)
    o.train_loader = types.SimpleNamespace(dataset=types.SimpleNamespace(tensors=[torch.zeros(3, n_features)]))
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    trial = study.ask()
    p = o.set_param_grid(trial)
    assert 2 <= p["width"] <= min(max(n_features - 1, 2), 1024)
