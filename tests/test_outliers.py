"""Tests for geogenie.outliers.detect_outliers (GeoGeneticOutlierDetector).

All data are small and synthetic. Coordinates are (longitude, latitude) in degrees.
"""

import logging
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from geogenie.outliers.detect_outliers import GeoGeneticOutlierDetector


def make_detector(gen, geo, n_jobs=1, verbose=0, args=None):
    """Build a detector without the plotting/basemap setup of __init__."""
    d = GeoGeneticOutlierDetector.__new__(GeoGeneticOutlierDetector)
    d.args = args if args is not None else SimpleNamespace(scale_factor=100)
    d.genetic_data = np.asarray(gen, dtype=float)
    d.geographic_data = np.asarray(geo, dtype=float)
    d.output_dir = "."
    d.prefix = "test"
    d.n_jobs = n_jobs
    d.seed = None
    d.url = None
    d.buffer = 0.1
    d.show_plots = False
    d.debug = False
    d.verbose = verbose
    d.logger = logging.getLogger("test_outliers")
    d.plot_gamma_dist = lambda *a, **k: None
    return d


def ibd_data(n=120, p=80, seed=0):
    """Genotypes (0/1/2) with strong clines over an Arkansas-sized area.

    Sample 0 is placed at the south-west corner but carries a genotype drawn
    from the north-east corner's allele frequencies (a planted genetic and
    geographic outlier). The last sample sits at the north-east corner.
    """
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-94.0, -90.0, n)
    lat = rng.uniform(33.0, 36.5, n)
    lon[0], lat[0] = -93.9, 33.1
    lon[-1], lat[-1] = -90.1, 36.4
    geo = np.c_[lon, lat]

    a = rng.normal(0, 2.0, p)
    b = rng.normal(0, 2.0, p)
    c = rng.normal(0, 0.3, p)

    def freqs(x_lon, y_lat):
        x = (np.asarray(x_lon) + 92.0) / 2.0
        y = (np.asarray(y_lat) - 34.75) / 1.75
        return 1 / (1 + np.exp(-(a * x[..., None] + b * y[..., None] + c)))

    gen = rng.binomial(2, freqs(lon, lat)).astype(float)
    gen[0] = rng.binomial(2, freqs(np.array(-90.1), np.array(36.4))).astype(float)
    return gen, geo


# --------------------------------------------------------------------------
# O1 / O2: geographic neighbour search in km, (lon, lat) handled, rows kept.
# --------------------------------------------------------------------------
def test_find_geo_knn_returns_great_circle_neighbours_in_km():
    lats = np.array([34.0, 34.1, 34.3, 34.6, 35.0, 34.0005])
    geo = np.c_[np.full(len(lats), -92.0), lats]
    d = make_detector(np.zeros((len(lats), 3)), geo)

    # Point 5 is ~56 m from point 0: excluded by min_nn_dist = 1 km.
    idx, dist = d.find_geo_knn(geo, 2, 1.0)

    assert idx.shape == (6, 2)
    np.testing.assert_array_equal(idx[0], [1, 2])
    deg_km = np.pi / 180 * 6371.0088
    np.testing.assert_allclose(dist[0], [0.1 * deg_km, 0.3 * deg_km], rtol=1e-6)
    # Row order is preserved: the last row's neighbours belong to the last row.
    np.testing.assert_array_equal(idx[5], [1, 2])
    assert np.all(idx >= 0)


def test_multi_stage_passes_min_nn_dist_in_km(monkeypatch):
    gen, geo = ibd_data(n=40, p=20)
    d = make_detector(gen, geo)
    seen = []
    orig = d.find_geo_knn

    def spy(coords, k, min_nn_dist):
        seen.append(min_nn_dist)
        return orig(coords, k, min_nn_dist)

    monkeypatch.setattr(d, "find_geo_knn", spy)
    d.multi_stage_outlier_knn(geo, gen, "composite", 0.05, 5, 2, 1000, 100)
    assert seen and all(m == pytest.approx(1.0) for m in seen)  # 1000 m


def test_predict_coords_knn_never_indexes_with_missing_neighbour():
    coords = np.array([[0.0, 0.0], [1.0, 2.0], [5.0, 5.0], [9.0, 9.0]])
    # -1 marks a missing neighbour; the index alone must exclude it, whatever
    # distance is stored next to it.
    idx = np.array([[1, -1], [-1, -1], [0, 2], [2, -1]])
    dist = np.array([[1.0, 0.5], [0.5, 0.5], [1.0, 1.0], [2.0, 0.5]])
    d = make_detector(np.zeros((4, 2)), coords)

    pred = d.predict_coords_knn(coords, dist, idx, 2)

    np.testing.assert_allclose(pred[0], coords[1])
    assert np.all(np.isnan(pred[1]))  # no valid neighbour -> no prediction
    np.testing.assert_allclose(pred[2], [2.5, 2.5])
    np.testing.assert_allclose(pred[3], coords[2])


# --------------------------------------------------------------------------
# O5: exact kNN, deterministic tie-break, independent of n_jobs.
# --------------------------------------------------------------------------
def test_find_gen_knn_is_exact_with_index_tie_break_for_any_n_jobs():
    rng = np.random.default_rng(1)
    gen = rng.integers(0, 3, size=(150, 8)).astype(float)  # many tied distances
    geo = np.c_[rng.uniform(-94, -90, 150), rng.uniform(33, 36, 150)]
    k = 10

    ref_d = cdist(gen, gen)
    np.fill_diagonal(ref_d, np.inf)
    ref_idx = np.argsort(ref_d, axis=1, kind="stable")[:, :k]

    for n_jobs in (1, 4):
        np.random.seed(n_jobs)
        d = make_detector(gen, geo, n_jobs=n_jobs)
        idx, dist = d.find_gen_knn(gen, k, 100)
        np.testing.assert_array_equal(idx, ref_idx)
        np.testing.assert_allclose(dist, np.take_along_axis(ref_d, ref_idx, 1))


# --------------------------------------------------------------------------
# O6: k neighbours, self excluded by index, exact duplicates, zero distance.
# --------------------------------------------------------------------------
def test_gen_knn_excludes_self_and_uses_duplicate_at_zero_distance():
    rng = np.random.default_rng(2)
    gen = rng.integers(0, 3, size=(30, 12)).astype(float)
    gen[1] = gen[0]  # exact genotype duplicates
    geo = np.c_[rng.uniform(-94, -90, 30), rng.uniform(33, 36, 30)]
    d = make_detector(gen, geo)
    k = 4

    idx, dist = d.find_gen_knn(gen, k, 100)

    assert idx.shape == (30, k)
    assert not np.any(idx == np.arange(30)[:, None])  # never self
    assert idx[0, 0] == 1 and idx[1, 0] == 0
    assert dist[0, 0] == 0 and dist[1, 0] == 0

    pred = d.predict_coords_knn(geo, dist, idx, 2)
    # The zero-distance duplicate determines the prediction, not the sample itself.
    np.testing.assert_allclose(pred[0], geo[1])
    np.testing.assert_allclose(pred[1], geo[0])


# --------------------------------------------------------------------------
# O3: multi-stage iteration continues until no new outliers are found.
# --------------------------------------------------------------------------
def test_run_multistage_iterates_until_no_new_outliers(monkeypatch, caplog):
    n = 10
    gen = np.zeros((n, 3))
    geo = np.c_[np.linspace(-94, -90, n), np.full(n, 34.0)]
    d = make_detector(gen, geo, verbose=1)
    script = [[0], [0], []]  # positions in the filtered arrays
    calls = []

    def fake_detect(g, x, optk, td, **kwargs):
        it = len(calls)
        calls.append(len(g))
        out = np.array(script[it], dtype=int)
        p = np.full(len(g), 0.5)
        p[out] = 0.001
        return td, out, np.ones(len(g)), p, 0.0, (1.0, 1.0)

    monkeypatch.setattr(d, "detect_outliers", fake_detect)
    with caplog.at_level(logging.INFO, logger="test_outliers"):
        _, flags, *_ = d.run_multistage(geo, gen, 0.05, 1.0, 100, n // 2, 2, "geographic", {})

    assert calls == [10, 9, 8]
    np.testing.assert_array_equal(np.where(flags)[0], [0, 1])
    msgs = [r.getMessage() for r in caplog.records]
    no_new = [i for i, m in enumerate(msgs) if "No new" in m]
    assert len(no_new) == 1
    assert "Iteration 3" in " ".join(msgs[: no_new[0]])


# --------------------------------------------------------------------------
# O7: R^2 of the KNN predictions.
# --------------------------------------------------------------------------
def test_r_squared_of_unrelated_predictions_is_not_one():
    rng = np.random.default_rng(3)
    actual = np.c_[rng.uniform(-94, -90, 50), rng.uniform(33, 36, 50)]
    unrelated = np.c_[rng.uniform(-94, -90, 50), rng.uniform(33, 36, 50)]
    d = make_detector(np.zeros((50, 2)), actual)

    *_, r2_bad = d.calculate_statistic(unrelated, actual, False, 1.0, 100)
    *_, r2_good = d.calculate_statistic(actual + 1e-4, actual, False, 1.0, 100)

    assert r2_bad < 0.5
    assert r2_good == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# O8: verbose >= 2, single analysis types, --scale_factor.
# --------------------------------------------------------------------------
def test_detect_outliers_with_verbose_2():
    gen, geo = ibd_data(n=40, p=20)
    d = make_detector(gen, geo, verbose=2)
    out = d.detect_outliers(geo, gen, 5, {}, analysis_type="geographic", min_nn_dist=1.0)
    assert len(out) == 6


@pytest.mark.parametrize("analysis_type", ["genetic", "geographic"])
def test_single_analysis_type(analysis_type):
    gen, geo = ibd_data(n=40, p=20)
    d = make_detector(gen, geo)
    geo_o, gen_o, res = d.multi_stage_outlier_knn(
        geo, gen, analysis_type, 0.05, 5, 2, 1000, 100
    )
    assert set(res) == {analysis_type}
    other = gen_o if analysis_type == "geographic" else geo_o
    assert other.size == 0


def test_composite_uses_args_scale_factor(monkeypatch):
    gen, geo = ibd_data(n=20, p=10)
    d = make_detector(gen, geo, args=SimpleNamespace(scale_factor=7))
    seen = {}

    def spy(*a, **k):
        seen.update(k)
        empty = np.array([], dtype=int)
        return empty, empty, {}

    monkeypatch.setattr(d, "multi_stage_outlier_knn", spy)
    d.composite_outlier_detection(sig_level=0.05, maxk=5, min_nn_dist=1000)
    assert seen["scale_factor"] == 7
    d.composite_outlier_detection(sig_level=0.05, maxk=5, min_nn_dist=1000, scale_factor=3)
    assert seen["scale_factor"] == 3


# --------------------------------------------------------------------------
# O9: optional Benjamini-Hochberg correction; default unchanged.
# --------------------------------------------------------------------------
def test_fit_gamma_mle_optional_bh_correction():
    rng = np.random.default_rng(4)
    D = np.r_[rng.gamma(4.0, 0.3, 300), [6.0, 8.0]]
    d = make_detector(np.zeros((2, 2)), np.zeros((2, 2)))

    raw, p, params = d.fit_gamma_mle(D, 0.05)
    raw_none, _, _ = d.fit_gamma_mle(D, 0.05, p_adjust=None)
    bh, p_bh, _ = d.fit_gamma_mle(D, 0.05, p_adjust="fdr_bh")

    np.testing.assert_array_equal(raw, np.where(p < 0.05)[0])
    np.testing.assert_array_equal(raw, raw_none)
    np.testing.assert_allclose(p, p_bh)  # returned p-values are unadjusted
    assert set(bh) <= set(raw) and len(bh) < len(raw)
    assert {300, 301} <= set(bh)
    with pytest.raises(ValueError):
        d.fit_gamma_mle(D, 0.05, p_adjust="bonferonni")


# --------------------------------------------------------------------------
# End to end: the genetic test works; results are positions and do not
# depend on n_jobs or the global random state.
# --------------------------------------------------------------------------
def test_composite_flags_planted_outlier_independent_of_n_jobs():
    gen, geo = ibd_data()
    results = []
    for n_jobs, seed in [(1, 0), (4, 0), (1, 99)]:
        np.random.seed(seed)
        d = make_detector(gen, geo, n_jobs=n_jobs)
        results.append(
            d.composite_outlier_detection(sig_level=0.05, maxk=10, min_nn_dist=1000)
        )

    first = results[0]
    assert set(first) == {"geographic", "genetic"}
    assert 0 in first["genetic"]
    assert 0 in first["geographic"]
    for r in results[1:]:
        for key in first:
            np.testing.assert_array_equal(r[key], first[key])
    for key in first:
        assert first[key].dtype.kind == "i"
        assert np.all((first[key] >= 0) & (first[key] < len(gen)))
