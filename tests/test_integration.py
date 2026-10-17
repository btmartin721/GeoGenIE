"""Tests for options and call sites that connect the modules."""

import sys
import types

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from geogenie.utils.argument_parser import setup_parser

BASE_ARGV = ["--vcf", "a.vcf", "--sample_data", "s.txt"]


def parse(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["geogenie"] + argv)
    return setup_parser()


def test_outlier_p_adjust_default_and_choice(monkeypatch):
    assert parse(monkeypatch, BASE_ARGV).outlier_p_adjust == "fdr_bh"
    assert parse(monkeypatch, BASE_ARGV + ["--outlier_p_adjust", "none"]).outlier_p_adjust == "none"


def test_bootstrap_method_and_oversample_cap_options(monkeypatch):
    a = parse(monkeypatch, BASE_ARGV)
    assert a.bootstrap_method == "samples" and a.max_oversample_factor == 5
    a = parse(monkeypatch, BASE_ARGV + ["--bootstrap_method", "loci", "--max_oversample_factor", "3"])
    assert a.bootstrap_method == "loci" and a.max_oversample_factor == 3


def test_local_shapefile_is_read_without_download(tmp_path, monkeypatch):
    from geogenie.utils import spatial_data_processors as sdp

    shp = tmp_path / "counties.gpkg"
    gpd.GeoDataFrame({"STATEFP": ["05"]}, geometry=[box(-94, 33, -90, 36)], crs="EPSG:4326").to_file(shp)

    def no_download(*a, **k):
        raise AssertionError("requests.get called for a local file")

    monkeypatch.setattr(sdp.requests, "get", no_download)
    p = sdp.SpatialDataProcessor.__new__(sdp.SpatialDataProcessor)
    p.output_dir = tmp_path / "out"
    p.logger = None
    p.crs = "EPSG:4326"
    gdf = p.extract_basemap_path_url(str(shp))
    assert len(gdf) == 1


@pytest.mark.parametrize("s2p,expected_n", [(None, 6), ("2", 2), ("s3, s5", 2), (["s1"], 1)])
def test_samples_to_plot_selection(s2p, expected_n, tmp_path):
    from geogenie.optimize.bootstrap import Bootstrap

    plotted = []
    coords = tmp_path / "coords.csv"
    pd.DataFrame({"sampleID": [f"s{i}" for i in range(6)], "x": -92.0, "y": 35.0}).to_csv(coords, index=False)
    b = Bootstrap.__new__(Bootstrap)
    b.args = types.SimpleNamespace(samples_to_plot=s2p, seed=1, highlight_basemap_counties=None, known_sample_data=str(coords), sample_data=str(coords))
    b.logger = None
    b.verbose = 0
    ids = [f"s{i}" for i in range(6)]
    df = pd.DataFrame({"sampleID": np.repeat(ids, 3), "x": np.random.default_rng(0).normal(-92, 0.1, 18),
                       "y": np.random.default_rng(1).normal(35, 0.1, 18)})

    class Proc:
        def to_geopandas(self, d):
            return d

        def calculate_statistics(self, gdf, known_coords=None):
            for sid, g in gdf.groupby("sampleID"):
                yield g, sid, None, {"sampleID": sid, "x_mean": g.x.mean(), "y_mean": g.y.mean()}

    b.plotting = types.SimpleNamespace(
        processor=Proc(), plot_sample_with_density=lambda group, sample_id, **k: plotted.append(sample_id)
    )
    b._grouped_ci_boot(df, dataset="pred")
    assert len(plotted) == expected_n
    if isinstance(s2p, str) and not s2p.isdigit():
        assert sorted(plotted) == ["s3", "s5"]
    if isinstance(s2p, list):
        assert plotted == ["s1"]


def test_optuna_plot_failure_does_not_stop_run(tmp_path):
    import logging

    import optuna

    from geogenie.optimize.optuna_opt import Optimize

    study = optuna.create_study()
    study.optimize(lambda t: t.suggest_float("lr", 1e-4, 1e-1), n_trials=2)
    o = Optimize.__new__(Optimize)
    o.verbose = 0
    o.logger = logging.getLogger("test")
    o.output_dir = str(tmp_path)
    o.prefix = "t"
    (tmp_path / "optimize").mkdir()

    def fail(study):
        raise ValueError("no kaleido")

    o.plotting = types.SimpleNamespace(make_optuna_plots=fail)
    o.write_optuna_study_details = lambda study: None
    o.process_optuna_results(study, study.best_trial)
    assert (tmp_path / "optimize" / "t_best_params.json").exists()
