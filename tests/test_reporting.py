"""Tests for bootstrap summary statistics and reported plots.

Covers SpatialDataProcessor.calculate_statistics and helpers, and PlotGenIE plots
(no network access: PlotGenIE is built without __init__ and given a synthetic basemap).
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from shapely.geometry import box

import geogenie.plotting.plotting as plotting_mod
from geogenie.plotting.plotting import PlotGenIE
from geogenie.utils.spatial_data_processors import SpatialDataProcessor


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture
def proc():
    return SpatialDataProcessor(logger=logging.getLogger("test"))


@pytest.fixture
def plotter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = PlotGenIE.__new__(PlotGenIE)
    p.device = "cpu"
    p.output_dir = str(tmp_path)
    p.prefix = "t"
    p.basemap_fips = "05"
    p.basemap_highlights = None
    p.show_plots = False
    p.fontsize = 10
    p.filetype = "png"
    p.dpi = 30
    p.remove_splines = False
    p.outbasepath = Path(tmp_path, "plots", "t.png")
    p.outbasepath.parent.mkdir(parents=True, exist_ok=True)
    p.logger = logging.getLogger("test_plot")
    p.processor = SpatialDataProcessor(
        output_dir=p.outbasepath.parent, logger=p.logger
    )
    # Synthetic "state": lon -94..-90, lat 33..36.5
    p.basemap = gpd.GeoDataFrame(
        {"STATEFP": ["05"]}, geometry=[box(-94.0, 33.0, -90.0, 36.5)], crs="EPSG:4326"
    )
    yield p
    plt.close("all")


def _stats(proc, df, **kw):
    return [r for _, _, _, r in proc.calculate_statistics(proc.to_geopandas(df), **kw)]


def _skewed(rng, sid="A", lon=-92.0, lat=35.0):
    xs = np.r_[lon + rng.normal(0, 0.05, 7), lon + 2.0, lon + 2.5, lon + 3.0]
    ys = np.r_[lat + rng.normal(0, 0.05, 7), lat + 1.0, lat + 1.2, lat + 1.5]
    return pd.DataFrame({"sampleID": sid, "x": xs, "y": ys})


# --------------------------------------------------------------------------- #
# Summary statistics (B3/S1, B4/S2, S3)
# --------------------------------------------------------------------------- #
def test_median_columns_are_per_axis_median(proc):
    df = _skewed(np.random.default_rng(0))
    r = _stats(proc, df)[0]
    assert r["x_median"] == pytest.approx(np.median(df.x))
    assert r["y_median"] == pytest.approx(np.median(df.y))
    assert abs(r["x_median"] - r["x_mean"]) > 0.5


def test_mean_counts_duplicate_replicates(proc):
    df = pd.DataFrame(
        {
            "sampleID": ["s"] * 5,
            "x": [-92.0, -92.0, -92.0, -92.0, -90.0],
            "y": [34.0, 34.0, 34.0, 34.0, 36.0],
        }
    )
    r = _stats(proc, df)[0]
    assert r["x_mean"] == pytest.approx(-91.6)
    assert r["y_mean"] == pytest.approx(34.4)


def test_spread_columns_values_and_units(proc):
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame(
        {"sampleID": "C", "x": -92 + rng.normal(0, 0.5, n), "y": 35 + rng.normal(0, 0.3, n)}
    )
    r = _stats(proc, df)[0]
    sdx, sdy = df.x.std(), df.y.std()
    assert r["std_dev_x"] == pytest.approx(sdx)
    assert r["std_dev_y"] == pytest.approx(sdy)
    assert r["q025_x"] == pytest.approx(np.percentile(df.x, 2.5))
    assert r["q975_x"] == pytest.approx(np.percentile(df.x, 97.5))
    assert r["q025_y"] == pytest.approx(np.percentile(df.y, 2.5))
    assert r["q975_y"] == pytest.approx(np.percentile(df.y, 97.5))
    assert r["half95_x"] == pytest.approx(1.96 * sdx)
    assert r["mean_ci95_x"] == pytest.approx(1.96 * sdx / np.sqrt(n))
    assert r["n_replicates"] == n

    mean = np.tile([df.x.mean(), df.y.mean()], (n, 1))
    d = proc.haversine_distance(df[["x", "y"]].to_numpy(), mean)
    assert r["drms_km"] == pytest.approx(np.sqrt(np.mean(d**2)))
    assert r["r95_km"] == pytest.approx(np.percentile(d, 95))
    # km, not degrees: ~0.5 deg lon at 35N ~ 45 km, 0.3 deg lat ~ 33 km
    assert 40 < r["drms_km"] < 75
    # 95% of replicates lie within r95_km of the mean
    assert np.mean(d <= r["r95_km"]) == pytest.approx(0.95, abs=0.01)


def test_misleading_columns_renamed(proc):
    df = _skewed(np.random.default_rng(2))
    r = _stats(proc, df)[0]
    for old in ("ci_95_x", "ci_95_y", "se_x", "se_y", "drms"):
        assert old not in r


def test_known_coords_error_uses_mean(proc):
    # Duplicated replicate predictions must count in the mean used for the error.
    df = pd.DataFrame(
        {"sampleID": "A", "x": [-92.0, -92.0, -92.0, -90.0], "y": [34.0, 34.0, 34.0, 36.0]}
    )
    known = pd.DataFrame({"sampleID": ["A"], "x": [-92.0], "y": [35.0]})
    r = _stats(proc, df, known_coords=known)[0]
    exp = proc.haversine_distance(
        np.array([[df.x.mean(), df.y.mean()]]), np.array([[-92.0, 35.0]])
    )[0]
    assert r["haversine_error"] == pytest.approx(exp)


# --------------------------------------------------------------------------- #
# S4 latent helpers
# --------------------------------------------------------------------------- #
def test_ensure_is_gdf_accepts_ndarray(proc):
    g = proc._ensure_is_gdf(np.array([[-92.0, 35.0], [-91.0, 34.0]]))
    assert isinstance(g, gpd.GeoDataFrame)
    assert list(g.geometry.x) == [-92.0, -91.0]


def test_spherical_mean_returns_lon_lat(proc):
    g = proc.to_geopandas(pd.DataFrame({"x": [-92.0, -92.0], "y": [34.0, 34.0]}))
    lon, lat = proc.spherical_mean(g)
    assert lon == pytest.approx(-92.0)
    assert lat == pytest.approx(34.0)


def test_nearest_neighbor_in_km(proc):
    g = proc.to_geopandas(pd.DataFrame({"x": [-92.0, -92.0, -80.0], "y": [34.0, 35.0, 34.0]}))
    nn = proc.nearest_neighbor(g)
    one_deg_lat = proc.haversine_distance(np.array([[-92.0, 34.0]]), np.array([[-92.0, 35.0]]))[0]
    assert nn[0] == pytest.approx(one_deg_lat, rel=1e-6)
    assert nn[1] == pytest.approx(one_deg_lat, rel=1e-6)


def test_detect_clusters_eps_in_km(proc):
    # Two points 5 km apart (lat) and one 300 km away; eps=10 km.
    g = proc.to_geopandas(
        pd.DataFrame({"x": [-92.0, -92.0, -92.0, -89.0], "y": [34.0, 34.045, 34.02, 34.0]})
    )
    labels = proc.detect_clusters(g, eps=10, min_samples=2)["cluster"].tolist()
    assert labels[:3] == [0, 0, 0] and labels[3] == -1


def test_detect_outliers_haversine(proc):
    # Longitude distance shrinks with latitude: at 80N a 2 deg lon offset is ~39 km,
    # smaller than a 1 deg lat offset (~111 km). Planar degrees would rank them the other way.
    g = proc.to_geopandas(
        pd.DataFrame({"x": [0.0] * 8 + [2.0, 0.0], "y": [80.0] * 8 + [80.0, 81.0]})
    )
    out = proc.detect_outliers(g, threshold=1)["is_outlier"].tolist()
    assert out[-1] is True or out[-1] == True  # noqa: E712
    assert not out[-2]


# --------------------------------------------------------------------------- #
# P1 / P4: per-sample bootstrap CI plot
# --------------------------------------------------------------------------- #
def _capture_density_plot(plotter, monkeypatch, df, **kw):
    cap = {"contour": [], "legend": None, "scatter": []}
    orig_contour = matplotlib.axes.Axes.contour
    orig_legend = matplotlib.axes.Axes.legend
    orig_scatter = matplotlib.axes.Axes.scatter

    def contour(self, *a, **k):
        cap["contour"].append((a, k))
        return orig_contour(self, *a, **k)

    def legend(self, *a, **k):
        cap["legend"] = a
        return orig_legend(self, *a, **k)

    def scatter(self, *a, **k):
        cap["scatter"].append((a, k))
        return orig_scatter(self, *a, **k)

    monkeypatch.setattr(matplotlib.axes.Axes, "contour", contour)
    monkeypatch.setattr(matplotlib.axes.Axes, "legend", legend)
    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", scatter)
    plotter.plot_sample_with_density(
        plotter.processor.to_geopandas(df), "S1", dataset="test", **kw
    )
    return cap


def test_bootstrap_contours_enclose_labelled_mass(plotter, monkeypatch):
    rng = np.random.default_rng(0)
    coords = np.c_[rng.normal(-92.3, 0.5, 100), rng.normal(34.1, 0.4, 100)]
    df = pd.DataFrame({"sampleID": "S1", "x": coords[:, 0], "y": coords[:, 1]})
    cap = _capture_density_plot(plotter, monkeypatch, df)

    (args, kwargs), = cap["contour"]
    xx, yy = args[0], args[1]
    levels, colors = list(kwargs["levels"]), list(kwargs["colors"])

    handles, labels = cap["legend"][0], cap["legend"][1]
    kde = stats.gaussian_kde(coords.T)
    dens = kde(kde.resample(100000, seed=1))
    found = 0
    for h, lab in zip(handles, labels):
        if not lab.endswith("% Density"):
            continue
        p = float(lab.split("%")[0]) / 100
        level = levels[colors.index(h.get_color())]
        assert np.mean(dens >= level) == pytest.approx(p, abs=0.03), lab
        found += 1
    assert found == 3
    # Grid extends beyond the bounding box of the points.
    assert xx.min() < coords[:, 0].min() and xx.max() > coords[:, 0].max()
    assert yy.min() < coords[:, 1].min() and yy.max() > coords[:, 1].max()


def test_bootstrap_plot_predicted_locality_is_unclipped_mean(plotter, monkeypatch):
    rng = np.random.default_rng(4)
    # Half of the replicates lie east of the basemap (lon > -90).
    coords = np.c_[np.r_[rng.normal(-91.0, 0.2, 10), rng.normal(-89.0, 0.2, 10)],
                   rng.normal(34.5, 0.2, 20)]
    df = pd.DataFrame({"sampleID": "S1", "x": coords[:, 0], "y": coords[:, 1]})
    cap = _capture_density_plot(plotter, monkeypatch, df)
    pts = [a for a, k in cap["scatter"] if k.get("label") == "Predicted Locality"]
    assert len(pts) == 1
    px, py = np.ravel(pts[0][0])[0], np.ravel(pts[0][1])[0]
    assert px == pytest.approx(coords[:, 0].mean())
    assert py == pytest.approx(coords[:, 1].mean())


# --------------------------------------------------------------------------- #
# P2: KDE-error regression plot
# --------------------------------------------------------------------------- #
def _capture_savefig(monkeypatch):
    figs = []
    orig = matplotlib.figure.Figure.savefig

    def savefig(self, *a, **k):
        figs.append(self)
        return orig(self, *a, **k)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", savefig)
    return figs


def test_density_regression_units_and_leave_one_out(plotter, monkeypatch):
    rng = np.random.default_rng(5)
    actual = np.c_[rng.uniform(-93, -91, 40), rng.uniform(34, 35.5, 40)]
    pred = actual + rng.normal(0, 0.2, actual.shape)
    figs = _capture_savefig(monkeypatch)
    plotter.polynomial_regression_plot(actual, pred, "test")

    ax = figs[-1].axes[0]
    pts = [c for c in ax.collections if len(c.get_offsets()) == len(actual)]
    xs = np.asarray(pts[0].get_offsets())[:, 0]
    # Expected: leave-one-out count within 25 km / (pi 25^2) per 1,000 km^2.
    P = plotter.processor
    exp = []
    for i in range(len(actual)):
        d = P.haversine_distance(np.tile(actual[i], (len(actual), 1)), actual)
        exp.append((np.sum(d <= 25.0) - 1) / (np.pi * 25.0**2) * 1000)
    np.testing.assert_allclose(xs, exp)
    assert "1,000" in ax.get_xlabel() and "25 km" in ax.get_xlabel()
    legend_text = " ".join(t.get_text() for t in ax.get_legend().get_texts())
    assert "County" not in legend_text and "Optimal" not in legend_text


def test_density_regression_reference_coords(plotter, monkeypatch):
    actual = np.array([[-92.0, 35.0], [-91.0, 34.0], [-92.5, 34.5], [-91.5, 35.2], [-92.2, 34.1]])
    ref = np.array([[-92.0, 35.05], [-92.0, 34.95], [-80.0, 30.0]])
    figs = _capture_savefig(monkeypatch)
    plotter.polynomial_regression_plot(actual, actual + 0.1, "test", reference_coords=ref)
    xs = figs[-1].axes[0].collections[0].get_offsets()[:, 0]
    assert xs[0] == pytest.approx(2 / (np.pi * 625) * 1000)
    assert np.all(xs[1:] == 0)
    assert "Training" in figs[-1].axes[0].get_xlabel()


# --------------------------------------------------------------------------- #
# P5: ECDF / z-score figures are closed
# --------------------------------------------------------------------------- #
def test_ecdf_and_zscore_close_figures(plotter):
    rng = np.random.default_rng(6)
    e = rng.gamma(2, 20, 100)
    plt.close("all")
    plotter.plot_cumulative_error_distribution(
        e, "t_test_cumulative_error_distribution.png",
        np.percentile(e, [25, 50, 75]), np.median(e), np.mean(e),
    )
    assert plt.get_fignums() == []
    plotter.plot_zscores((e - e.mean()) / e.std(), "t_test_zscores.png")
    assert plt.get_fignums() == []


# --------------------------------------------------------------------------- #
# P6: geographic error map
# --------------------------------------------------------------------------- #
def test_error_map_labels_and_normalized_centroids(plotter, monkeypatch):
    rng = np.random.default_rng(7)
    actual = np.c_[rng.uniform(-93, -91, 30), rng.uniform(34, 35.5, 30)]
    pred = actual + rng.normal(0, 0.2, actual.shape)
    figs = _capture_savefig(monkeypatch)
    normalized_centroids = np.array([[0.24, 0.69], [0.25, 0.70]])
    plotter.plot_geographic_error_distribution(
        actual, pred, "test", centroids=normalized_centroids, n_contour_levels=5
    )
    ax = figs[-1].axes[0]
    assert ax.get_xlabel() == "Longitude"
    assert ax.get_ylabel() == "Latitude"
    labels = [c.get_label() for c in ax.collections]
    assert not any(l.startswith("Centroids") for l in labels)


# --------------------------------------------------------------------------- #
# P7: target distribution plot in degrees
# --------------------------------------------------------------------------- #
class _Scaler:
    def inverse_transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.c_[X[:, 0] * 4 - 94, X[:, 1] * 3.5 + 33]


def test_target_distributions_inverse_transformed(plotter, monkeypatch):
    rng = np.random.default_rng(8)
    tr, va, te = (rng.uniform(0, 1, (20, 2)) for _ in range(3))
    figs = _capture_savefig(monkeypatch)
    plotter.plot_data_distributions(tr, va, te, is_target=True, target_scaler=_Scaler())
    ax_lon, ax_lat = figs[-1].axes[:2]
    lo, hi = ax_lon.get_xlim()
    assert -95 < lo and hi < -89
    lo, hi = ax_lat.get_xlim()
    assert 32 < lo and hi < 37.5


# --------------------------------------------------------------------------- #
# P8: unknown samples not drawn as outliers
# --------------------------------------------------------------------------- #
def test_outlier_plot_excludes_unknowns(plotter, monkeypatch):
    seen = {}
    orig = plotting_mod.sns.scatterplot

    def scatterplot(*a, **k):
        seen["data"] = k["data"].copy()
        return orig(*a, **k)

    monkeypatch.setattr(plotting_mod.sns, "scatterplot", scatterplot)
    locs = np.array([[-92.0, 35.0], [-91.0, 34.0], [-92.5, 34.5], [-91.5, 35.2]])
    mask = np.array([True, False, True, False])  # 1 = outlier, 3 = unknown
    known = np.array([True, True, True, False])
    plotter.plot_outliers(mask, locs, known_mask=known)
    d = seen["data"]
    assert len(d) == 3
    assert (d["Outliers"] == "True").sum() == 1


# --------------------------------------------------------------------------- #
# P10: labels
# --------------------------------------------------------------------------- #
def test_metric_labels_pearson_r(plotter):
    df = pd.DataFrame({"Metric": ["pearson_corr_longitude", "pearson_corr_latitude"]})
    out = plotter.update_metric_labels(df)
    assert not out["Metric"].str.contains("R^2", regex=False).any()
    assert out["Metric"].str.contains("Pearson's r").all()


def test_bootstrap_aggregate_labels(plotter, monkeypatch):
    rng = np.random.default_rng(9)
    n = 20
    df = pd.DataFrame(
        {
            "mean_dist": rng.normal(50, 5, n),
            "median_dist": rng.normal(40, 5, n),
            "pearson_corr_longitude": rng.uniform(0.6, 0.9, n),
            "pearson_corr_latitude": rng.uniform(0.6, 0.9, n),
            "spearman_corr_longitude": rng.uniform(0.6, 0.9, n),
            "spearman_corr_latitude": rng.uniform(0.6, 0.9, n),
        }
    )
    figs = _capture_savefig(monkeypatch)
    plotter.plot_bootstrap_aggregates(df)
    texts = [
        t.get_text()
        for f in figs
        for a in f.axes
        if a.get_legend() is not None
        for t in a.get_legend().get_texts()
    ]
    assert not any("R^2" in t for t in texts)
    assert any("Pearson's r" in t for t in texts)


def test_error_distribution_and_gamma_labels(plotter, monkeypatch):
    rng = np.random.default_rng(10)
    figs = _capture_savefig(monkeypatch)
    plotter.plot_error_distribution(rng.gamma(2, 20, 100), "t_test_error_distributions")
    ax0 = figs[-1].axes[0]
    assert ax0.get_ylabel() == "Probability Density"
    assert "Sampling" not in ax0.get_title()

    Dg = rng.gamma(3, 2, 200)
    plotter.plot_gamma_distribution(3.0, 2.0, Dg, 0.05, "t_gamma_geographic", "Gamma")
    ax = figs[-1].axes[0]
    assert "β" not in ax.get_xlabel() and "scale=" in ax.get_xlabel()
    # Observed D histogram is drawn with the fitted pdf.
    assert len(ax.patches) >= 50


# --------------------------------------------------------------------------- #
# S13: oversample visualisation
# --------------------------------------------------------------------------- #
def test_oversample_plot_length_check(plotter):
    arr = np.array([[-92.0, 35.0], [-91.0, 34.0], [-92.5, 34.5]])
    plotter.visualize_oversample_clusters(
        arr, np.array([0, 1, 1]), ["original", "original", "synthetic"]
    )
    with pytest.raises(ValueError, match="arr, cluster_labels, and sample_origin_list"):
        plotter.visualize_oversample_clusters(
            arr, np.array([0, 1]), ["original", "original", "synthetic"]
        )
