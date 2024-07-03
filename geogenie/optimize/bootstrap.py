import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from torch import optim
from torch.utils.data import DataLoader

from geogenie.plotting.plotting import PlotGenIE
from geogenie.samplers.interpolate import run_genotype_interpolator
from geogenie.utils.callbacks import callback_init
from geogenie.utils.data import CustomDataset, UnlabeledDataset


class Bootstrap:
    def __init__(
        self,
        train_loader,
        val_loader,
        test_loader,
        val_indices,
        test_indices,
        pred_indices,
        sample_data,
        samples,
        args,
        ds,
        best_params,
        weighted_sampler,
        device,
    ):
        """Class to run model with bootstrapping to estimate validation error.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
            val_indices (np.ndarray): Indices for validation data.
            test_indices (np.ndarray): Indices for the test data.
            pred_indices (np.ndarray): Indices for the unknown samples.
            sample_data (pd.DataFrame): Sample IDs and coordinate data.
            samples (np.ndarray): All sample IDs.
            args (argparse.Namespace): User-supplied arguments.
            ds (DataStructure): DataStructure instance that stores data and metadata for features and labels.
            best_params (dict): Best parameters from parameter search, or if optimization was not run, then best_params represents user-supplied arguments.
            weighted_sampler (GeographicDensitySampler): Weighted sampler to use for probabalistic sampling.
            device (torch.device): Device to run the model on {'cpu' or 'cuda'}.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.val_indices = val_indices
        self.test_indices = test_indices
        self.pred_indices = pred_indices
        self.sample_data = sample_data
        self.samples = samples
        self.args = args
        self.ds = ds
        self.best_params = best_params
        self.weighted_sampler = weighted_sampler
        self.nboots = args.nboots
        self.verbose = self.args.verbose
        self.device = device
        self.dtype = torch.float32 if args.dtype == "float32" else torch.float64

        self.logger = logging.getLogger(__name__)

        self.plotting = PlotGenIE(
            device,
            self.args.output_dir,
            self.args.prefix,
            self.args.basemap_fips,
            self.args.highlight_basemap_counties,
            self.args.shapefile,
            show_plots=self.args.show_plots,
            fontsize=self.args.fontsize,
            filetype=self.args.filetype,
            dpi=self.args.plot_dpi,
            remove_splines=self.args.remove_splines,
        )

    def _resample_loaders(self, train_loader, val_loader, test_loader, pred_loader):
        """
        Resample only the validation and test data loaders using bootstrapping.

        Args:
            train_loader: DataLoader for the training dataset (to be resampled).
            val_loader: DataLoader for the validation dataset (to be resampled).
            test_loader: DataLoader for the test dataset (to be resampled).
            pred_loader: DataLoader for the pred dataset (to be resampled).

        Returns:
            Tuple containing the original train loader and resampled validation and test loaders.
        """
        train_loader, resampled_indices = self._resample_boot(
            train_loader, None, is_val=False
        )
        val_loader, _ = self._resample_boot(val_loader, resampled_indices, is_val=True)
        test_loader, __ = self._resample_boot(
            test_loader, resampled_indices, is_val=True
        )
        pred_loader, ___ = self._resample_boot(
            pred_loader, resampled_indices, is_val=False, is_pred=True
        )

        return (train_loader, val_loader, test_loader, pred_loader, resampled_indices)

    def _resample_boot(
        self, loader, sampled_feature_indices=None, is_val=False, is_pred=False
    ):
        """
        Apply resampling to a given DataLoader.

        Args:
            loader (torch.utils.data.DataLoader): The DataLoader to resample.
            sampled_feature_indices (np.ndarray): Numpy array of feature indices to use if is_val is True. Defaults to None.
            is_val (bool): If True, then it's a validation or test dataset. Otherwise, it's a training dataset.
            is_pred (bool): If True, then it's the unknown pred dataset.

        Returns:
            DataLoader: The resampled DataLoader.
        """
        dataset = loader.dataset
        features = dataset.features.numpy().copy()

        if not is_pred:
            labels = dataset.labels.numpy()
            sample_weights = dataset.sample_weights.numpy()

        if sampled_feature_indices is None or not is_val:
            if not is_pred:
                sample_size = int(self.args.feature_prop * features.shape[1])
                sampled_feature_indices = np.random.choice(
                    np.arange(features.shape[1]), size=sample_size, replace=True
                )

        if sampled_feature_indices is None:
            raise TypeError(
                "sampled_feature_indices was not set correctly; got NoneType."
            )

        features = features[:, sampled_feature_indices]

        use_sampler = {"sampler", "both"}
        shuffle = (
            not is_val and not self.args.use_weighted in use_sampler and not is_pred
        )
        kwargs = {"batch_size": loader.batch_size, "shuffle": shuffle}

        if is_pred:
            dataset = UnlabeledDataset(features)
        else:
            dataset = CustomDataset(
                features, labels, sample_weights=sample_weights, dtype=self.dtype
            )

        return DataLoader(dataset, **kwargs), sampled_feature_indices

    def train_one_bootstrap(self, boot, ModelClass, train_func):
        self.logger.info(f"Entered train_one_bootstrap with boot: {boot}")
        try:
            if self.args.verbose >= 1:
                self.logger.info(
                    f"Starting bootstrap iteration {boot + 1}/{self.nboots}"
                )

            X_pred = self.ds.data["X_pred"].copy()
            pred_tensor = torch.tensor(X_pred, dtype=self.dtype, device=self.device)
            pred_dataset = UnlabeledDataset(pred_tensor)
            pred_loader = DataLoader(pred_dataset)

            (
                train_loader,
                val_loader,
                test_loader,
                pred_loader_resamp,
                resampled_indices,
            ) = self._resample_loaders(
                self.train_loader, self.val_loader, self.test_loader, pred_loader
            )

            model = ModelClass(
                len(resampled_indices),
                width=self.best_params["width"],
                nlayers=self.best_params["nlayers"],
                dropout_prop=self.best_params["dropout_prop"],
                device=self.device,
                output_width=train_loader.dataset.labels.shape[1],
                dtype=self.dtype,
            ).to(self.device)

            optimizer = self.extract_best_params(self.best_params, model)

            if self.args.verbose >= 2:
                self.logger.info(f"Creating EarlyStopping with boot: {boot}")

            early_stop, lr_scheduler = callback_init(
                optimizer, self.args, trial=None, boot=boot
            )

            trained_model = train_func(
                train_loader,
                val_loader,
                model,
                optimizer,
                trial=None,
                objective_mode=False,
                do_bootstrap=True,
                early_stopping=early_stop,
                lr_scheduler=lr_scheduler,
            )

            if self.args.verbose >= 2:
                self.logger.info(f"Completed bootstrap training with replicate: {boot}")

            return (
                trained_model,
                resampled_indices,
                val_loader,
                test_loader,
                pred_loader_resamp,
            )

        except Exception as e:
            self.logger.error(
                f"Error during model training in bootstrap iteration {boot}: {str(e)}"
            )
            raise e

    def bootstrap_training_generator(self, ModelClass, train_func):
        n_jobs = os.cpu_count() if self.args.n_jobs == -1 else self.args.n_jobs

        if self.args.verbose >= 1:
            self.logger.info(
                f"Multiprocessing: Using {n_jobs} threads for bootstrapping..."
            )

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(
                    self.train_one_bootstrap, boot, ModelClass, train_func
                ): boot
                for boot in range(self.nboots)
            }

            for future in futures:
                boot = futures[future]

                if self.args.verbose >= 2:
                    self.logger.info(f"Awaiting result for boot {boot}")
                try:
                    result = future.result()

                    if self.args.verbose >= 2:
                        self.logger.info(f"Completed job for boot {boot}")
                    if all(r is None for r in result):
                        self.logger.error(f"Bootstrap iteration {boot} had an error.")
                        continue
                    yield result
                except Exception as exc:
                    self.logger.error(
                        f"Bootstrap iteration {boot} generated an exception: {exc}"
                    )
                    raise exc

    def extract_best_params(self, best_params, model):
        lr = best_params["lr"] if "lr" in best_params else best_params["learning_rate"]

        l2 = (
            best_params["l2_weight"]
            if "l2_weight" in best_params
            else best_params["l2_reg"]
        )

        # Define the criterion and optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
        return optimizer

    def _set_loader_kwargs(self, resampled_weights, kwargs, is_val):
        if self.args.use_weighted == "none":
            kwargs["shuffle"] = True if not is_val else False
        else:
            if self.args.oversample_method == "none":
                if self.args.use_weighted in {"sampler", "both"}:
                    kwargs["sampler"] = deepcopy(self.weighted_sampler)

                if self.args.use_weighted in {"loss", "both"}:
                    if self.args.use_weighted != "both":
                        kwargs["shuffle"] = True if not is_val else False
        if not isinstance(resampled_weights, torch.Tensor):
            resampled_weights = torch.tensor(resampled_weights, dtype=self.dtype)
        return resampled_weights, kwargs

    def save_bootstrap_results(
        self, boot, test_preds, test_indices, write_func, dataset, test_metrics=None
    ):
        """
        Save the results of each bootstrap iteration.

        Args:
            boot (int): The current bootstrap iteration.
            test_preds (np.array): Predictions made by the model in the current iteration.
            test_indices (np.ndarray): Indices for current validation or test set.
            write_func (callable): Function to write the predictions to file.
            dataset (str): Which dataset to use. Valid options: {"test", "val"}.
            test_metrics (dict, optional): Test set metrics for the current iteration. Should be None if dataset == 'pred', otherwise should be defined. Defaults to None.

        Returns:
            pd.DataFrame: Output predictions.
        """
        if dataset not in {"test", "val", "pred"}:
            msg = f"dataset must be 'test', 'val', or 'pred', but got {dataset}"
            self.logger.error(msg)
            raise ValueError(msg)

        outdir = Path(self.args.output_dir)

        if dataset != "pred":
            if test_metrics is None:
                msg = "'test_metrics' cannot be NoneType if dataset != 'pred'"
                self.logger.error(msg)
                raise TypeError(msg)
            pth = outdir / "bootstrap_metrics" / dataset
            pth.mkdir(exist_ok=True, parents=True)
            of = f"{self.args.prefix}_bootrep{boot}_{dataset}_metrics.json"
            boot_file_path = pth / of

            with open(boot_file_path, "w") as fout:
                json.dump(test_metrics, fout)

            # If this is the first bootstrap iteration, delete existing file
            if boot_file_path.exists():
                if self.verbose >= 2:
                    self.logger.warn("Found existing metrics file. Removing it.")
                boot_file_path.unlink()  # Remove the file.

        if test_metrics is not None and dataset == "pred":
            msg = "'test_metrics' was defined for unknown predictions."
            self.logger.error(msg)
            raise TypeError(msg)

        ds = "unknown" if dataset == "pred" else dataset
        pth = outdir / "bootstrap_predictions" / ds
        pth.mkdir(exist_ok=True, parents=True)
        of = f"{self.args.prefix}_bootrep{boot}_{ds}_predictions.csv"
        outfile = pth / of

        if isinstance(test_preds, dict):
            test_preds = np.array(list(test_preds.values()))

        return write_func(test_preds, test_indices, outfile)

    def perform_bootstrap_training(self, train_func, pred_func, write_func, ModelClass):
        if self.verbose >= 1:
            self.logger.info("Starting bootstrap training.")

        outdir = self.args.output_dir

        if self.args.oversample_method != "none":
            try:
                (self.train_loader, _, __, ___, _____) = run_genotype_interpolator(
                    self.train_loader, self.args, self.ds, self.dtype, self.plotting
                )
            except Exception as e:
                msg = f"Unexpected error occurred during genotype interpolation prior to bootstrapping: {str(e)}"
                self.logger.error(msg)
                raise e

        bootstrap_preds, bootstrap_test_preds, bootstrap_val_preds = [], [], []
        bootstrap_test_metrics, bootstrap_val_metrics = [], []

        for boot, (
            model,
            resampled_indices,
            val_loader,
            test_loader,
            pred_loader,
        ) in enumerate(self.bootstrap_training_generator(ModelClass, train_func)):
            if self.verbose >= 1:
                self.logger.info(f"Processing bootstrap {boot + 1}/{self.nboots}")

            outpth = Path(outdir) / "models"
            bootrep_file = outpth / f"{self.args.prefix}_model_bootrep{boot}.pt"

            if isinstance(model, tuple):
                model = model[0]

            if model is None:
                msg = f"Model was not trained successfully for bootstrap {boot}"
                self.logger.error(msg)
                raise TypeError(msg)

            val_preds, val_metrics = pred_func(
                model,
                val_loader,
                None,
                return_truths=False,
                use_rf=self.args.use_gradient_boosting,
                bootstrap=True,
            )

            test_preds, test_metrics = pred_func(
                model,
                test_loader,
                None,
                return_truths=False,
                use_rf=self.args.use_gradient_boosting,
                bootstrap=True,
            )

            pred_loader_resampled = DataLoader(
                CustomDataset(
                    self.ds.data["X_pred"][:, resampled_indices].copy(),
                    sample_weights=None,
                    dtype=self.dtype,
                ),
                batch_size=pred_loader.batch_size,
                shuffle=False,
            )

            real_preds = pred_func(
                model,
                pred_loader_resampled,
                None,
                return_truths=False,
                use_rf=self.args.use_gradient_boosting,
                bootstrap=True,
                is_val=False,
            )

            test_preds_df, test_sample_data = self._extract_sample_ids(
                test_preds, self.samples[self.test_indices]
            )

            val_preds_df, val_sample_data = self._extract_sample_ids(
                val_preds, self.samples[self.val_indices]
            )

            real_preds_df, pred_sample_data = self._extract_sample_ids(
                real_preds, self.ds.all_samples[self.pred_indices]
            )

            test_preds = dict(zip(test_sample_data, test_preds))
            val_preds = dict(zip(val_sample_data, val_preds))
            real_preds = dict(zip(pred_sample_data, real_preds))

            bootstrap_preds.append(real_preds_df)
            bootstrap_test_preds.append(test_preds_df)
            bootstrap_val_preds.append(val_preds_df)
            bootstrap_test_metrics.append(test_metrics)
            bootstrap_val_metrics.append(val_metrics)

            self.save_bootstrap_results(
                boot,
                test_preds,
                self.test_indices,
                write_func,
                "test",
                test_metrics=test_metrics,
            )
            self.save_bootstrap_results(
                boot,
                val_preds,
                self.val_indices,
                write_func,
                "val",
                test_metrics=val_metrics,
            )
            self.save_bootstrap_results(
                boot,
                real_preds,
                self.pred_indices,
                write_func,
                "pred",
                test_metrics=None,
            )

            torch.save(model.state_dict(), bootrep_file)

        boot_real_df = self._process_boot_preds(outdir, bootstrap_preds, dataset="pred")
        boot_test_df = self._process_boot_preds(
            outdir, bootstrap_test_preds, dataset="test"
        )
        boot_val_df = self._process_boot_preds(
            outdir, bootstrap_val_preds, dataset="val"
        )

        self.boot_real_df_ = boot_real_df
        self.boot_test_df_ = boot_test_df
        self.boot_val_df_ = boot_val_df

        metrics = [bootstrap_test_metrics, bootstrap_val_metrics]
        dfs = {}
        for d, m in zip(["test", "val"], metrics):
            fn = f"{self.args.prefix}_bootstrap_{d}_metrics.csv"
            dfs[d] = self._bootrep_metrics_to_csv(outdir, fn, m, d)
        self.boot_test_metrics_df_ = dfs["test"]
        self.boot_val_metrics_df_ = dfs["val"]

        self.plotting.plot_bootstrap_aggregates(
            pd.DataFrame.from_dict(bootstrap_test_metrics)
        )

        if self.verbose >= 1:
            self.logger.info("Bootstrap training completed!")

    def _extract_sample_ids(self, preds, samples):
        df = pd.DataFrame(preds, columns=["x", "y"])
        df["sampleID"] = samples
        df = df[["sampleID", "x", "y"]]
        return df, df["sampleID"].to_numpy().tolist()

    def _process_boot_preds(self, outdir, bootstrap_preds, dataset):
        """Process bootstrap predictions on real unseen data and write the summarized predictions to file.

        Args:
            outdir (str): Output directory to use.
            bootstrap_pred (list of pd.DataFrame): Data to process.
            dataset (str): Which dataset to use {"val", "test", "pred"}.
        """
        if dataset not in {"val", "test", "pred"}:
            msg = f"dataset must be either 'val', 'test', or 'pred': {dataset}"
            self.logger.error(msg)
            raise ValueError(msg)

        bootstrap_df = pd.concat(bootstrap_preds)
        predout = self._grouped_ci_boot(bootstrap_df, dataset)

        pth = Path(outdir)
        pth = pth.joinpath("bootstrap_summaries")
        pth.mkdir(exist_ok=True, parents=True)
        of = f"{self.args.prefix}_bootstrap_summary_{dataset}_predictions.csv"
        summary_outfile = pth.joinpath(of)

        predout.to_csv(summary_outfile, header=True, index=False)
        return predout

    def _process_boot_preds(self, outdir, bootstrap_preds, dataset):
        if dataset not in {"val", "test", "pred"}:
            msg = f"dataset must be either 'val', 'test', or 'pred': {dataset}"
            self.logger.error(msg)
            raise ValueError(msg)

        bootstrap_df = pd.concat(bootstrap_preds)
        predout = self._grouped_ci_boot(bootstrap_df, dataset)

        pth = Path(outdir)
        pth = pth.joinpath("bootstrap_summaries")
        pth.mkdir(exist_ok=True, parents=True)
        of = f"{self.args.prefix}_bootstrap_summary_{dataset}_predictions.csv"
        summary_outfile = pth.joinpath(of)

        predout.to_csv(summary_outfile, header=True, index=False)
        return predout

    def _grouped_ci_boot(self, df, dataset):
        if self.args.known_sample_data is not None and dataset != "pred":
            df_known = pd.read_csv(
                self.args.known_sample_data,
                names=["sampleID", "x", "y"],
                sep="\t",
                header=0,
            )
        else:
            df_known = None

        if df_known is None and dataset != "pred":
            self.logger.warning("Known coordinates were not provided.")

        results = []

        n_uniq_samples = len(df["sampleID"].unique())

        if self.args.samples_to_plot is None:
            plot_indices = np.arange(n_uniq_samples)
        elif (
            self.args.samples_to_plot is not None
            and self.args.samples_to_plot.isdigit()
        ):
            plot_indices = np.random.choice(
                np.arange(n_uniq_samples),
                size=int(self.args.samples_to_plot),
                replace=False,
            )
        else:
            df = df.copy()
            s2p = self.args.samples_to_plot
            if not isinstance(s2p, str):
                msg = f"'--samples_to_plot' must be of type str, but got: {type(s2p)}"
                self.logger.error(msg)
                raise TypeError(msg)
            sids = s2p.split(",")
            sids = [x.strip() for x in sids]
            plot_indices = np.where(np.isin(df["sampleID"].unique(), sids))[0]

        gdf = self.plotting.processor.to_geopandas(df)

        for i, (group, sample_id, dfk, resd) in enumerate(
            self.plotting.processor.calculate_statistics(gdf, known_coords=df_known)
        ):
            if i in plot_indices:
                self.plotting.plot_sample_with_density(
                    group,
                    sample_id,
                    df_known=dfk,
                    dataset=dataset,
                    gray_counties=self.args.highlight_basemap_counties,
                )

            results.append(resd)

        dfres = pd.DataFrame(results)

        if df_known is not None:
            dfres = dfres.sort_values(by="sampleID")
            df_known = df_known[~df_known["x"].isna()]
            df_known = df_known[df_known["sampleID"].isin(dfres["sampleID"])]
            df_known = df_known.sort_values(by="sampleID")

            self.plotting.plot_geographic_error_distribution(
                df_known[["x", "y"]].to_numpy(),
                dfres[["x_mean", "y_mean"]].to_numpy(),
                dataset,
                buffer=self.args.bbox_buffer,
                marker_scale_factor=self.args.sample_point_scale,
                min_colorscale=self.args.min_colorscale,
                max_colorscale=self.args.max_colorscale,
                n_contour_levels=self.args.n_contour_levels,
            )

            self.plotting.polynomial_regression_plot(
                df_known[["x", "y"]].to_numpy(),
                dfres[["x_mean", "y_mean"]].to_numpy(),
                dataset,
                dtype=self.dtype,
            )

        return dfres

    def _bootrep_metrics_to_csv(self, outdir, outfile, bootstrap_res, dataset):
        """Write bootstrap replicates (rows) containing evaluation metrics (columns) to a CSV file.

        Args:
            outdir (str): Output directory to use.
            outfile (str): output filename to use.
            bootstrap_res (list of dictionaries): Results to load into pd.DataFrame.
            dataset (str): Dataset to use. Should be one of {"test", "val", "pred"}.
        """
        df = pd.DataFrame.from_dict(bootstrap_res)
        pth = Path(outdir)
        pth = pth.joinpath("bootstrap_metrics", dataset)
        pth.mkdir(exist_ok=True, parents=True)
        boot_outfile = pth.joinpath(outfile)

        df.to_csv(boot_outfile, header=True, index=True)
        return df
