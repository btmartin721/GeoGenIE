import json
import logging
import os
from copy import deepcopy
from pathlib import Path

import scipy.stats as stats
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from torch import optim
from torch.utils.data import DataLoader

from geogenie.plotting.plotting import PlotGenIE
from geogenie.samplers.interpolate import run_genotype_interpolator
from geogenie.utils.callbacks import callback_init
from geogenie.utils.data import CustomDataset
from geogenie.utils.scorers import haversine_numpy, compute_drms


class Bootstrap:
    def __init__(
        self,
        train_loader,
        val_loader,
        test_loader,
        val_indices,
        test_indices,
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
            val_indices (np.ndarray): Indices for validation data.
            test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
            test_indices (np.ndarray): Indices for the test data.
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

        self.subdir = "bootstrap"
        self.logger = logging.getLogger(__name__)

        self.plotting = PlotGenIE(
            device,
            self.args.output_dir,
            self.args.prefix,
            self.args.basemap_fips,
            self.args.highlight_basemap_counties,
            show_plots=self.args.show_plots,
            fontsize=self.args.fontsize,
            filetype=self.args.filetype,
            dpi=self.args.plot_dpi,
        )

    def bootstrap_training_generator(self, ModelClass, train_func):
        """Generator for training models on bootstrapped samples.

        Args:
            ModelClass (torch.nn.Module): PyTorch model to train.
            train_func (callable): Callable function with PyTorch training loop.

        Yields:
            Trained model for each bootstrap sample.
            Training losses.
            Validation losses.
            Total train time.
        """
        for boot in range(self.nboots):
            if self.args.verbose >= 1:
                self.logger.info(
                    f"Starting bootstrap iteration {boot + 1}/{self.nboots}"
                )

            (
                train_loader,
                val_loader,
                test_loader,
                resampled_indices,
            ) = self._resample_loaders(
                self.train_loader, self.val_loader, self.test_loader
            )

            # Reinitialize the model and optimizer each bootstrap
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
            early_stop, lr_scheduler = callback_init(optimizer, self.args)

            try:
                # Train the model.
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
            except Exception as e:
                self.logger.error(
                    f"Error during model training in bootstrap iteration {boot + 1}: {str(e)}",
                )
                raise e

            yield trained_model, resampled_indices, val_loader, test_loader

    def _resample_loaders(self, train_loader, val_loader, test_loader):
        """
        Resample the data loaders using bootstrapping.

        Args:
            train_loader: DataLoader for the training dataset.
            val_loader: DataLoader for the validation dataset.
            test_loader: DataLoader for the test dataset.

        Returns:
            Tuple containing the resampled train, validation, and test loaders.
        """
        num_features = train_loader.dataset.features.size(1)

        # Generating resampled indices using torch.randint
        resampled_indices = torch.randint(
            num_features, (num_features,), dtype=torch.long
        )

        train_loader = self._resample_boot(resampled_indices, train_loader)
        val_loader = self._resample_boot(resampled_indices, val_loader)
        test_loader = self._resample_boot(resampled_indices, test_loader)

        return train_loader, val_loader, test_loader, resampled_indices

    def _resample_boot(self, resampled_indices, loader):
        """
        Apply resampling to a given DataLoader.

        Args:
            resampled_indices (torch.Tensor): The indices to sample from.
            loader: The DataLoader to resample.

        Returns:
            DataLoader: The resampled DataLoader.
        """
        features = loader.dataset.features.detach().clone()
        resampled_features = features[:, resampled_indices]
        resampled_features = resampled_features.to(dtype=self.dtype)

        labels = loader.dataset.labels.detach().clone()
        sample_weights = loader.dataset.sample_weights.detach().clone()

        kwargs = {"batch_size": self.train_loader.batch_size}
        sample_weights, kwargs = self._set_loader_kwargs(sample_weights, kwargs)

        dataset = CustomDataset(
            resampled_features, labels, sample_weights=sample_weights, dtype=self.dtype
        )
        loader = DataLoader(dataset, **kwargs)
        return loader

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

    def _set_loader_kwargs(self, resampled_weights, kwargs):
        if self.args.use_weighted == "none":
            kwargs["shuffle"] = True
        else:
            if self.args.use_weighted in {"sampler", "both"}:
                kwargs["sampler"] = deepcopy(self.weighted_sampler)

            if self.args.use_weighted in {"loss", "both"}:
                if not isinstance(resampled_weights, torch.Tensor):
                    resampled_weights = torch.tensor(
                        resampled_weights, dtype=self.dtype
                    )

                if self.args.use_weighted != "both":
                    kwargs["shuffle"] = True
        return resampled_weights, kwargs

    def save_bootstrap_results(
        self, boot, test_metrics, test_preds, write_func, dataset
    ):
        """
        Save the results of each bootstrap iteration.

        Args:
            boot (int): The current bootstrap iteration.
            test_metrics (dict): Test set metrics for the current iteration.
            test_preds (np.array): Predictions made by the model in the current iteration.
            write_func (callable): Function to write the predictions to file.
            dataset (str): Which dataset to use. Valid options: {"test", "val"}.

        Returns:
            pd.DataFrame: Output predictions.
        """
        if dataset not in {"test", "val"}:
            msg = f"'dataset' must be either 'test' or 'val', but got: {dataset}"
            self.logger.error(msg)
            raise ValueError(msg)

        pth = Path(self.args.output_dir)
        pth = pth.joinpath("bootstrap_metrics", dataset)
        pth.mkdir(exist_ok=True, parents=True)
        of = f"{self.args.prefix}_bootrep{boot}_{dataset}_metrics.json"
        boot_file_path = pth.joinpath(of)

        with open(boot_file_path, "w") as fout:
            json.dump(test_metrics, fout)

        # If this is the first bootstrap iteration, delete existing file
        if boot == 0 and os.path.exists(boot_file_path):
            if self.verbose >= 2:
                self.logger.warn("Found existing metrics file. Removing it.")
            os.remove(boot_file_path)

        pth = Path(self.args.output_dir)
        pth = pth.joinpath("bootstrap_predictions", dataset)
        pth.mkdir(exist_ok=True, parents=True)
        of = f"{self.args.prefix}_bootrep{boot}_{dataset}_predictions.csv"
        outfile = pth.joinpath(of)

        if isinstance(test_preds, dict):
            test_preds = np.array(list(test_preds.values()))

        return write_func(
            test_preds,
            self.test_indices,
            self.sample_data,
            outfile,
        )

    def perform_bootstrap_training(
        self, train_func, pred_func, write_func, unseen_pred_func, ModelClass
    ):
        """
        Perform training using bootstrap resampling to obtain confidence interval estimates.

        Args:
            train_func (callable): Callable PyTorch training loop.
            pred_func (callable): Callable predict function.
            write_func (callable): Callable function to write predictions to file.
            unseen_pred_func (callable): Callable to make unseen predictions and write to file.
            ModelClass (callable): Callable for subclassed PyTorch model.

        Returns:
            None
        """
        if self.verbose >= 1:
            self.logger.info("Starting bootstrap training.")

        outdir = self.args.output_dir

        if self.args.oversample_method != "none":
            try:
                # Do this only once, before bootstrapping.
                (self.train_loader, _, __, ___, _____) = run_genotype_interpolator(
                    self.train_loader, self.args, self.ds, self.dtype
                )
            except Exception as e:
                msg = f"Unexpected error occurred during genotype interpolation prior to bootstrapping: {str(e)}"
                self.logger.error(msg)
                raise e

        bootstrap_gen = self.bootstrap_training_generator(ModelClass, train_func)

        bootstrap_preds, bootstrap_test_preds, bootstrap_val_preds = [], [], []
        bootstrap_test_metrics, bootstrap_val_metrics = [], []

        # Generator function. Processes one bootstrap at a time.
        for boot, (model, resampled_indices, val_loader, test_loader) in enumerate(
            bootstrap_gen
        ):
            if self.verbose >= 1:
                self.logger.info(
                    f"Processing bootstrap {boot + 1}/{self.nboots}",
                )

            bootrep_file = os.path.join(
                outdir, "models", f"{self.args.prefix}_model_bootrep{boot}.pt"
            )

            # Save or evaluate the trained model
            torch.save(model.state_dict(), bootrep_file)

            # Process predictions for each bootstrap replicate.
            val_preds, val_metrics = pred_func(
                model,
                val_loader,
                None,
                return_truths=False,
                use_rf=False,
                bootstrap=True,
            )
            # Process predictions for each bootstrap replicate.
            test_preds, test_metrics = pred_func(
                model,
                test_loader,
                None,
                return_truths=False,
                use_rf=False,
                bootstrap=True,
            )

            test_preds_df, test_sample_data = self._extract_sample_ids(
                test_preds, self.test_indices
            )

            val_preds_df, val_sample_data = self._extract_sample_ids(
                val_preds, self.val_indices
            )

            test_preds = dict(zip(test_sample_data, test_preds))
            val_preds = dict(zip(val_sample_data, val_preds))

            # Writes predictions to separate files for each bootrep.
            real_preds_df = unseen_pred_func(
                model,
                self.device,
                use_rf=False,
                col_indices=resampled_indices,
                boot_rep=boot,
            )

            # Get bootstraps of real predictions.
            bootstrap_preds.append(real_preds_df)

            # Get metrics for test dataset.
            bootstrap_test_metrics.append(test_metrics)

            # Save test set predictions to list of dataframes.
            bootstrap_test_preds.append(test_preds_df)

            # Get validation set predictions and metrics into list of dfs.
            bootstrap_val_preds.append(val_preds_df)
            bootstrap_val_metrics.append(val_metrics)

            # Save metrics and predictions
            self.save_bootstrap_results(
                boot, test_metrics, test_preds, write_func, "test"
            )
            self.save_bootstrap_results(boot, val_metrics, val_preds, write_func, "val")

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

        # Additional processing like plotting aggregate results, computing
        # statistics, etc.
        self.plotting.plot_bootstrap_aggregates(
            pd.DataFrame.from_dict(bootstrap_test_metrics)
        )

        if self.verbose >= 1:
            self.logger.info("Bootstrap training completed!")

    def _extract_sample_ids(self, preds, indices):
        df = pd.DataFrame(preds, columns=["x", "y"])
        df["sampleID"] = self.samples[indices]
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

    def _grouped_ci_boot(self, df, dataset):
        """
        Process locality data for each sample to calculate mean, standard deviation, confidence intervals, and DRMS (Distance Root Mean Square).

        Args:
            df (pd.DataFrame): DataFrame containing 'x', 'y' coordinates and 'sampleID'.
            dataset (str): Which dataset is being used. Should be one of {"test", "val", "pred"}.

        Returns:
            pd.DataFrame: DataFrame with calculated statistics for each sample.
        """
        results = []

        for sample_id, group in df.groupby("sampleID"):
            mean_x, mean_y = group["x"].mean(), group["y"].mean()
            std_dev_x, std_dev_y = group["x"].std(), group["y"].std()
            n = len(group)

            ci_95_x = 1.96 * std_dev_x / np.sqrt(n)
            ci_95_y = 1.96 * std_dev_y / np.sqrt(n)

            se_x = 1.96 * std_dev_x
            se_y = 1.96 * std_dev_y
            drms = compute_drms(std_dev_x, std_dev_y)

            resd = {
                "sampleID": sample_id,
                "x_mean": mean_x,
                "y_mean": mean_y,
                "std_dev_x": std_dev_x,
                "std_dev_y": std_dev_y,
                "ci_95_x": ci_95_x,
                "ci_95_y": ci_95_y,
                "se_x": se_x,
                "se_y": se_y,
                "drms": drms,
            }

            results.append(resd)

            if dataset == "pred" and sample_id in self.args.samples_to_plot:
                self.plotting.plot_boot_ci_data(
                    group,
                    resd,
                    sample_id,
                    self.args.shapefile,
                    self.args.known_sample_data,
                )

        return pd.DataFrame(results)

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
