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
