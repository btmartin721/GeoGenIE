"""Tests for the GeoGenIE training / prediction core (geogenie.py, loss.py,
models.py, callbacks.py, scorers.py, argument_parser.py)."""

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import optuna
import pytest
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from geogenie.geogenie import GeoGenIE
from geogenie.models.models import MLPRegressor
from geogenie.utils.argument_parser import setup_parser
from geogenie.utils.callbacks import EarlyStopping, callback_init
from geogenie.utils.data import CustomDataset
from geogenie.utils.loss import WeightedDRMSLoss, WeightedHuberLoss, weighted_rmse_loss
from geogenie.utils.scorers import calculate_r2_knn, kstest
from geogenie.utils.spatial_data_processors import SpatialDataProcessor

torch.set_num_threads(2)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
class LinearNorm:
    """Stand-in target scaler: normalised [0, 1] -> degrees in a small box."""

    lo = np.array([-94.0, 33.0])
    span = np.array([4.0, 3.0])

    def inverse_transform(self, y):
        return np.asarray(y) * self.span + self.lo

    def transform(self, y):
        return (np.asarray(y) - self.lo) / self.span


def make_args(tmp_path, **kw):
    args = dict(
        oversample_method="none",
        max_epochs=40,
        grad_clip=False,
        verbose=0,
        debug=False,
        output_dir=str(tmp_path),
        prefix="t",
        early_stop_patience=1000,
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=1000,
        filetype="png",
        bbox_buffer=0.1,
        sample_point_scale=2,
        min_colorscale=0,
        max_colorscale=300,
        n_contour_levels=5,
        batch_size=16,
    )
    args.update(kw)
    return SimpleNamespace(**args)


def make_genie(tmp_path, dtype=torch.float32, criterion=weighted_rmse_loss, **kw):
    g = GeoGenIE.__new__(GeoGenIE)
    g.args = make_args(tmp_path, **kw)
    g.dtype = dtype
    g.logger = logging.getLogger("test_core")
    g.criterion = criterion
    g.device = "cpu"
    g.boot = None
    g.plotting = mock.MagicMock()
    g.plotting.processor = SpatialDataProcessor(output_dir=None, logger=g.logger)
    return g


def make_data(n=120, p=50, seed=1):
    gen = torch.Generator().manual_seed(seed)
    X = torch.randn(n, p, generator=gen)
    W = torch.randn(p, 2, generator=gen) * 0.05
    y = torch.sigmoid(X @ W)
    return X, y


def loaders(X, y, n_train=80, batch_size=16):
    tr = DataLoader(
        CustomDataset(X[:n_train], y[:n_train]), batch_size=batch_size, shuffle=True
    )
    va = DataLoader(
        CustomDataset(
            X[n_train:], y[n_train:], sample_ids=[f"v{i}" for i in range(len(X) - n_train)]
        ),
        batch_size=batch_size,
    )
    return tr, va


def parse(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["geogenie"] + argv)
    return setup_parser()


BASE_ARGV = ["--vcf", "a.vcf", "--sample_data", "s.txt"]


# --------------------------------------------------------------------------
# C2 / B8: best checkpoint restored when max_epochs is reached
# --------------------------------------------------------------------------
def _train_noisy(tmp_path, **train_kw):
    torch.manual_seed(1)
    np.random.seed(1)
    g = make_genie(tmp_path, max_epochs=train_kw.get("max_epochs", 35))
    X, y = make_data()
    tr, va = loaders(X, y)
    model = MLPRegressor(X.shape[1], width=32, nlayers=4, dropout_prop=0.25)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)  # noisy val curve
    return g, tr, va, model, opt


def test_best_checkpoint_restored_when_max_epochs_reached(tmp_path):
    g, tr, va, model, opt = _train_noisy(tmp_path)
    m, _, val_losses, _ = g.train_model(tr, va, model, opt)
    best = min(val_losses[1:])
    assert best < val_losses[-1]  # the last epoch is not the best one
    final, _ = g.test_step(va, m)
    assert final == pytest.approx(best, rel=1e-5)


def test_best_checkpoint_restored_in_bootstrap_mode(tmp_path):
    g, tr, va, model, opt = _train_noisy(tmp_path)
    es, lrs = callback_init(opt, g.args, trial=None, boot=0)
    m, _ = g.train_model(
        tr, va, model, opt, do_bootstrap=True, early_stopping=es, lr_scheduler=lrs
    )
    final, _ = g.test_step(va, m)
    assert final == pytest.approx(es.best_score, rel=1e-5)


# --------------------------------------------------------------------------
# O1 / C12: Optuna trial.report is called each epoch
# --------------------------------------------------------------------------
def test_trial_reports_intermediate_values(tmp_path):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    g, tr, va, model, opt = _train_noisy(tmp_path, max_epochs=5)
    g.args.max_epochs = 5
    study = optuna.create_study(pruner=optuna.pruners.NopPruner())

    def objective(trial):
        es, lrs = callback_init(opt, g.args, trial=trial)
        g.train_model(
            tr,
            va,
            model,
            opt,
            trial=trial,
            objective_mode=True,
            early_stopping=es,
            lr_scheduler=lrs,
        )
        return 0.0

    study.optimize(objective, n_trials=1)
    assert len(study.trials[0].intermediate_values) == 5


# --------------------------------------------------------------------------
# C3: config file vs CLI precedence and validation
# --------------------------------------------------------------------------
def _write_cfg(tmp_path, text):
    p = tmp_path / "c.yaml"
    p.write_text(text)
    return str(p)


def test_cli_overrides_config(tmp_path, monkeypatch):
    cfg = _write_cfg(
        tmp_path,
        "vcf: a.vcf\nsample_data: s.txt\nmax_epochs: 5000\nearly_stop_patience: 100\nbatch_size: 8\n",
    )
    a = parse(monkeypatch, ["--config", cfg, "--max_epochs", "10", "--early_stop_patience", "5"])
    assert a.max_epochs == 10
    assert a.early_stop_patience == 5
    assert a.batch_size == 8  # config still replaces the default


def test_config_values_are_type_converted(tmp_path, monkeypatch):
    cfg = _write_cfg(
        tmp_path,
        "vcf: a.vcf\nsample_data: s.txt\nlearning_rate: 1e-3\ncriterion: RMSE\n"
        "do_bootstrap: 'false'\ngrad_clip: 'true'\nsamples_to_plot: [A, B]\n",
    )
    a = parse(monkeypatch, ["--config", cfg])
    assert isinstance(a.learning_rate, float) and a.learning_rate == pytest.approx(1e-3)
    assert a.criterion == "rmse"
    assert a.do_bootstrap is False
    assert a.grad_clip is True
    assert a.samples_to_plot == ["A", "B"]


def test_invalid_config_value_rejected(tmp_path, monkeypatch):
    cfg = _write_cfg(tmp_path, "vcf: a.vcf\nsample_data: s.txt\nmax_epochs: -3\n")
    with pytest.raises(SystemExit):
        parse(monkeypatch, ["--config", cfg])


# --------------------------------------------------------------------------
# C15 / C16: CLI validators, --factor
# --------------------------------------------------------------------------
def test_n_components_is_int(monkeypatch):
    a = parse(monkeypatch, BASE_ARGV + ["--n_components", "2", "--embedding_type", "tsne"])
    assert a.n_components == 2


def test_zero_l2_and_dropout_accepted(monkeypatch):
    a = parse(monkeypatch, BASE_ARGV + ["--l2_reg", "0", "--dropout_prop", "0"])
    assert a.l2_reg == 0.0 and a.dropout_prop == 0.0
    with pytest.raises(SystemExit):
        parse(monkeypatch, BASE_ARGV + ["--dropout_prop", "1"])


def test_factor_warns_that_it_is_ignored(monkeypatch):
    with pytest.warns(UserWarning, match="factor"):
        parse(monkeypatch, BASE_ARGV + ["--factor", "0.5"])


# --------------------------------------------------------------------------
# C4 / C5: losses
# --------------------------------------------------------------------------
def test_drms_is_euclidean_in_normalised_space():
    preds = torch.zeros(2, 2)
    targets = torch.tensor([[0.3, 0.4], [0.0, 0.0]])
    loss = WeightedDRMSLoss()(preds, targets)
    assert loss.item() == pytest.approx(np.sqrt(0.25 / 2), rel=1e-6)
    w = torch.tensor([3.0, 1.0])
    loss_w = WeightedDRMSLoss()(preds, targets, sample_weight=w)
    assert loss_w.item() == pytest.approx(np.sqrt(3 * 0.25 / 4), rel=1e-6)


def test_huber_has_no_target_smoothing():
    y = torch.rand(16, 2)
    crit = WeightedHuberLoss(delta=0.5)
    assert crit(y.clone(), y).item() == pytest.approx(0.0, abs=1e-12)


def test_huber_explicit_smoothing_factor_ignored():
    y = torch.rand(16, 2)
    with pytest.warns(UserWarning):
        crit = WeightedHuberLoss(delta=0.5, smoothing_factor=0.1)
    assert crit(y.clone(), y).item() == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# #10: BatchNorm with batch size 1
# --------------------------------------------------------------------------
def test_batchnorm_applied_in_eval_for_single_sample():
    torch.manual_seed(0)
    model = MLPRegressor(10, width=8, nlayers=2, dropout_prop=0.0)
    model.train()
    model(torch.randn(64, 10) * 3 + 5)  # update BN running statistics
    model.eval()
    X = torch.randn(4, 10) * 3 + 5
    with torch.no_grad():
        batch_out = model(X)
        single_out = model(X[:1])
    assert torch.allclose(single_out, batch_out[:1], atol=1e-6)


def test_train_step_drops_single_sample_batch(tmp_path):
    torch.manual_seed(0)
    g = make_genie(tmp_path)
    X, y = make_data(n=17, p=10)
    tr = DataLoader(CustomDataset(X, y), batch_size=16, shuffle=False)
    model = MLPRegressor(10, width=8, nlayers=2, dropout_prop=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = g.train_step(tr, model, opt, grad_clip=False, objective_mode=False)
    assert np.isfinite(loss)
    steps = {int(st["step"]) for st in opt.state.values()}
    assert steps == {1}  # the batch of 1 was not used for an update


# --------------------------------------------------------------------------
# C8: --dtype float64 with float32 datasets
# --------------------------------------------------------------------------
def test_float64_model_with_float32_dataset(tmp_path):
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        g = make_genie(tmp_path, dtype=torch.float64, max_epochs=2)
        X, y = make_data(n=40, p=10)
        tr, va = loaders(X.float(), y.float(), n_train=30)
        model = MLPRegressor(10, width=8, nlayers=2, dtype=torch.float64)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        m, tl, vl, _ = g.train_model(tr, va, model, opt)
        assert all(np.isfinite(tl)) and all(np.isfinite(vl))
    finally:
        torch.set_default_dtype(old)


# --------------------------------------------------------------------------
# C13: non-finite validation loss in EarlyStopping
# --------------------------------------------------------------------------
def test_early_stopping_ignores_nan(tmp_path):
    es = EarlyStopping(str(tmp_path), "t", patience=10)
    model = torch.nn.Linear(2, 2)
    for v in [1.0, 0.5, float("nan"), 0.9]:
        es(v, model)
    assert es.best_score == 0.5
    assert es.val_loss_min == 0.5
    assert es.counter == 2


# --------------------------------------------------------------------------
# C14: validation loss over the whole set, not averaged per batch
# --------------------------------------------------------------------------
def test_val_loss_independent_of_batch_size(tmp_path):
    torch.manual_seed(0)
    g = make_genie(tmp_path)
    X, y = make_data(n=33, p=10)
    model = MLPRegressor(10, width=8, nlayers=2, dropout_prop=0.0)
    model.eval()
    with torch.no_grad():
        exact = weighted_rmse_loss(y, model(X)).item()
    for bs in (32, 11, 33):
        va = DataLoader(CustomDataset(X, y, sample_ids=list(range(33))), batch_size=bs)
        v, _ = g.test_step(va, model)
        assert v == pytest.approx(exact, rel=1e-5)


# --------------------------------------------------------------------------
# C6: rho metric
# --------------------------------------------------------------------------
def test_rho_is_per_axis_not_pooled(tmp_path):
    g = make_genie(tmp_path)
    rng = np.random.default_rng(0)
    gt = np.c_[rng.uniform(-94, -90, 200), rng.uniform(33, 36, 200)]
    pred = np.c_[rng.uniform(-94, -90, 200), rng.uniform(33, 36, 200)]  # no skill
    _, values, _ = g.get_all_stats(
        pred,
        gt,
        lambda d: np.median(np.abs(d - np.median(d))),
        lambda d: np.std(d) / np.mean(d),
        lambda d, t: np.mean(d < t),
    )
    m = g._create_metrics_dictionary(values)
    lon = spearmanr(pred[:, 0], gt[:, 0])[0]
    lat = spearmanr(pred[:, 1], gt[:, 1])[0]
    assert m["rho"] == pytest.approx(np.mean([lon, lat]))
    assert abs(m["rho"]) < 0.2


# --------------------------------------------------------------------------
# C9, C10, C11, C7: evaluate_and_save_results
# --------------------------------------------------------------------------
def _eval_setup(tmp_path):
    torch.manual_seed(0)
    g = make_genie(tmp_path)
    for d in ("validation", "test", "training", "plots"):
        (tmp_path / d).mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    n_real, n_syn, p = 40, 25, 10
    X = torch.tensor(rng.normal(size=(n_real, p)), dtype=torch.float32)
    y = torch.tensor(rng.uniform(0.1, 0.9, size=(n_real, 2)), dtype=torch.float32)
    Xv = torch.tensor(rng.normal(size=(20, p)), dtype=torch.float32)
    yv = torch.tensor(rng.uniform(0.1, 0.9, size=(20, 2)), dtype=torch.float32)
    real_train = DataLoader(CustomDataset(X, y), batch_size=16, shuffle=True)
    syn = DataLoader(
        CustomDataset(
            torch.cat([X, X[:n_syn]]), torch.cat([y, y[:n_syn]])
        ),
        batch_size=16,
        shuffle=True,
    )
    val = DataLoader(
        CustomDataset(Xv, yv, sample_ids=[f"v{i}" for i in range(20)]), batch_size=16
    )
    g.ds = SimpleNamespace(
        train_loader=real_train, val_loader=val, test_loader=val, norm=LinearNorm()
    )
    g.train_loader_interp = syn
    model = MLPRegressor(p, width=8, nlayers=2, dropout_prop=0.0)
    model.train()
    model(X)
    model.eval()
    return g, model, n_real, y


def test_train_metrics_and_plot_use_real_samples_in_degrees(tmp_path):
    g, model, n_real, y = _eval_setup(tmp_path)
    g.evaluate_and_save_results(model, [1.0], [1.0], dataset="val")
    calls = [
        c for c in g.plotting.polynomial_regression_plot.call_args_list if c.args[2] == "train"
    ]
    assert len(calls) == 1
    actual, preds = calls[0].args[0], calls[0].args[1]
    assert len(actual) == n_real == len(preds)
    # Degrees (not normalised), and the same set of samples as the real labels.
    expected = LinearNorm().inverse_transform(y.numpy())
    assert np.allclose(np.sort(actual[:, 0]), np.sort(expected[:, 0]), atol=1e-4)


def test_evaluate_does_not_run_full_train_forward_pass(tmp_path):
    g, model, n_real, _ = _eval_setup(tmp_path)
    sizes = []
    model.register_forward_hook(lambda m, i, o: sizes.append(i[0].shape[0]))
    g.evaluate_and_save_results(model, [1.0], [1.0], dataset="val")
    # Only loader batches (<= batch size); no separate pass over all of X_train.
    assert max(sizes) <= 16


def test_val_and_test_plot_names(tmp_path):
    g, model, _, _ = _eval_setup(tmp_path)
    g.evaluate_and_save_results(model, [1.0], [1.0], dataset="val")
    g.evaluate_and_save_results(model, [1.0], [1.0], dataset="test")
    cum = [c.args[1] for c in g.plotting.plot_cumulative_error_distribution.call_args_list]
    zs = [c.args[1] for c in g.plotting.plot_zscores.call_args_list]
    assert cum == [
        "t_validation_cumulative_error_distribution.png",
        "t_test_cumulative_error_distribution.png",
    ]
    assert zs == ["t_validation_zscores.png", "t_test_zscores.png"]


def test_oversampling_does_not_mutate_ds_train_loader(tmp_path, monkeypatch):
    g = make_genie(tmp_path, oversample_method="kmeans", max_epochs=2)
    X, y = make_data(n=60, p=10)
    tr, va = loaders(X, y, n_train=40)
    g.ds = SimpleNamespace(train_loader=tr)
    Xs, ys = torch.cat([X[:40], X[:20]]), torch.cat([y[:40], y[:20]])
    syn = DataLoader(CustomDataset(Xs, ys), batch_size=16, shuffle=True)

    def fake_interp(train_loader, args, ds, dtype, plotting):
        return syn, None, Xs.numpy(), ys.numpy(), np.ones(len(Xs))

    monkeypatch.setattr("geogenie.geogenie.run_genotype_interpolator", fake_interp)
    model = MLPRegressor(10, width=8, nlayers=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g.train_model(tr, va, model, opt)
    assert len(g.ds.train_loader.dataset) == 40
    assert g.train_loader_interp is syn


# --------------------------------------------------------------------------
# C18: final model saved under the updated prefix
# --------------------------------------------------------------------------
def test_final_model_uses_updated_prefix(tmp_path):
    g = make_genie(tmp_path)
    g.args = SimpleNamespace(
        **vars(g.args),
        seed=1,
        gpu_number=None,
        criterion="rmse",
        use_gradient_boosting=False,
        do_gridsearch=False,
        load_best_params=None,
        do_bootstrap=False,
        learning_rate=1e-3,
        l2_reg=0.0,
    )
    (tmp_path / "benchmarking").mkdir()

    def fake_load():
        g.args.prefix = "t_N10_L5"
        g.ds = SimpleNamespace(
            define_params=lambda a: None,
            params={"learning_rate": 1e-3, "l2_reg": 0.0},
            train_loader=None,
            val_loader=None,
        )

    g.load_data = fake_load
    g.perform_standard_training = lambda *a, **k: (torch.nn.Linear(1, 1), [], [], None)
    g.evaluate_and_save_results = lambda *a, **k: None
    g.make_unseen_predictions = lambda *a, **k: None
    saved = []
    g.save_model = lambda model, fn: saved.append(Path(fn).name)
    g.train_test_predict()
    assert saved == ["t_N10_L5_trained_model.pt"]


# --------------------------------------------------------------------------
# C19: tuned lr / l2_weight vs learning_rate / l2_reg
# --------------------------------------------------------------------------
def test_tuned_params_win_and_are_logged(tmp_path, caplog):
    g = make_genie(tmp_path)
    params = {"lr": 1e-3, "learning_rate": 5e-3, "l2_weight": 0.0, "l2_reg": 1e-5}
    with caplog.at_level(logging.INFO, logger="test_core"):
        lr, l2 = g.resolve_optimizer_params(params)
    assert (lr, l2) == (1e-3, 0.0)
    assert "overrides learning_rate=0.005" in caplog.text
    assert "overrides l2_reg=1e-05" in caplog.text


# --------------------------------------------------------------------------
# C20 / C21: scorers
# --------------------------------------------------------------------------
def test_r2_knn_unrelated_data_near_zero():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(200, 2)), rng.normal(size=(200, 2))
    assert calculate_r2_knn(a, b) < 0.05
    assert calculate_r2_knn(a, a) == pytest.approx(1.0)


def test_kstest_uses_lilliefors():
    from statsmodels.stats.diagnostic import lilliefors

    rng = np.random.default_rng(3)
    gt = np.c_[rng.uniform(-94, -90, 100), rng.uniform(33, 36, 100)]
    pred = gt + rng.normal(0, 0.3, size=gt.shape)
    ks, p, _ = kstest(gt, pred)
    errs = SpatialDataProcessor(output_dir=None, logger=logging.getLogger("x")).haversine_distance(gt, pred)
    ks_ref, p_ref = lilliefors(np.asarray(errs, dtype=float), dist="norm", pvalmethod="table")
    assert ks == pytest.approx(ks_ref)
    assert p == pytest.approx(p_ref)
