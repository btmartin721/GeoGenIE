import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader

from geogenie.plotting.plotting import PlotGenIE
from geogenie.samplers.interpolate import run_genotype_interpolator
from geogenie.utils.callbacks import callback_init
from geogenie.utils.data import CustomDataset
from geogenie.utils.exceptions import InvalidInputShapeError
from geogenie.utils.utils import check_column_dtype, read_csv_with_dynamic_sep


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
        device,
    ):
        """Class to run model with bootstrapping to estimate validation error.

        This class runs the model with bootstrapping to estimate the validation error. The model is trained for each bootstrap iteration and the predictions are saved to file. The class also generates summary statistics and confidence intervals for the bootstrapped predictions.

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
        self.nboots = args.nboots
        self.verbose = self.args.verbose
        self.device = device
        self.dtype = torch.float32 if args.dtype == "float32" else torch.float64

        self.logger = logging.getLogger(__name__)

        self.plotting = PlotGenIE(
            device,
            self.args.output_dir,
            self.args.prefix + "_bootstrap",
            self.args.basemap_fips,
            self.args.highlight_basemap_counties,
            self.args.shapefile,
            show_plots=self.args.show_plots,
            fontsize=self.args.fontsize,
            filetype=self.args.filetype,
            dpi=self.args.plot_dpi,
            remove_splines=self.args.remove_splines,
        )

        self.thread_local = threading.local()

    def _get_thread_local_rng(self, replicate_seed=None):
        """Get a thread-local random generator with a unique seed.

        Args:
            replicate_seed (int, optional): Seed for the random number generator. Defaults to None.

        Returns:
            np.random.Generator: Random number generator with a unique seed.
        """
        if not hasattr(self.thread_local, "rng"):
            self.thread_local.rng = np.random.default_rng(replicate_seed)
        return self.thread_local.rng

    def _resample_loaders(
        self, train_loader, val_loader, test_loader, X_pred, replicate_seed=None
    ):
        """Resample the train, test, and validation data loaders using bootstrapping.

        This method resamples the train, test, and validation data loaders using bootstrapping. The data loaders are resampled with replacement per columns (loci) and the number of features to sample is determined by the feature proportion set by the user. If the feature proportion is not set, then the number of features to sample is equal to the number of features in the dataset.

        Args:
            train_loader: DataLoader for the training dataset (to be resampled).
            val_loader: DataLoader for the validation dataset (to be resampled).
            test_loader: DataLoader for the test dataset (to be resampled).
            X_pred: numpy array with the pred dataset (to be resampled).
            replicate_seed: Seed for the random number generator.

        Returns:
            Tuple containing the original train loader and resampled validation, test, and unknown pred loaders.
        """
        train_loader, resampled_indices = self._resample_boot(
            train_loader, None, is_val=False
        )
        val_loader, _ = self._resample_boot(
            val_loader, resampled_indices, is_val=True, replicate_seed=replicate_seed
        )
        test_loader, __ = self._resample_boot(
            test_loader, resampled_indices, is_val=True, replicate_seed=replicate_seed
        )
        pred_loader, ___ = self._resample_boot(
            X_pred,
            resampled_indices,
            is_val=False,
            is_pred=True,
            replicate_seed=replicate_seed,
        )

        return (train_loader, val_loader, test_loader, pred_loader)

    def _resample_boot(
        self,
        loader,
        sampled_feature_indices=None,
        is_val=False,
        is_pred=False,
        replicate_seed=None,
    ):
        """Resamples the data loader with replacement for bootstrapping per columns (loci).

        This method resamples the data loader with replacement for bootstrapping per columns (loci). The number of features to sample is determined by the feature proportion set by the user. If the feature proportion is not set, then the number of features to sample is equal to the number of features in the dataset.

        Args:
            loader: DataLoader to resample.
            sampled_feature_indices: Indices of the features to sample.
            is_val: Whether the data loader is for the validation set.
            is_pred: Whether the data loader is for the prediction set.
            replicate_seed: Seed for the random number generator.

        Returns:
            DataLoader: Resampled DataLoader

        Raises:
            TypeError: If the loader is not a DataLoader or numpy array.
            TypeError: If the sampled_feature_indices is not set correctly.

        """
        # Make a thread-local copy of the data
        if isinstance(loader, np.ndarray):
            features = loader.copy()
        elif isinstance(loader, DataLoader):
            dataset = deepcopy(loader.dataset)
            features = dataset.features.numpy().copy()
        else:
            msg = f"Invalid type passed to _resample_loaders(): {type(loader)}"
            self.logger.error(msg)
            raise TypeError(msg)

        # Use thread-local RNG for sampling with unique seed
        rng = self._get_thread_local_rng(replicate_seed)

        if sampled_feature_indices is None and not is_val and not is_pred:
            sample_size = int(features.shape[1])
            idx = np.arange(features.shape[1])
            sampled_feature_indices = rng.choice(idx, size=sample_size, replace=True)

        if sampled_feature_indices is None:
            msg = "sampled_feature_indices was not set correctly."
            self.logger.error(msg)
            raise TypeError(msg)

        features = features[:, sampled_feature_indices]

        use_sampler = {"sampler", "both"}
        shuffle = not (is_pred or is_val or self.args.use_weighted in use_sampler)

        kwargs = {}
        if is_pred:
            kwargs["labels"] = None
            kwargs["sample_weights"] = None
            kwargs["sample_ids"] = self.ds.data["pred_samples"]
        else:
            kwargs["labels"] = dataset.labels
            kwargs["sample_weights"] = dataset.sample_weights
            kwargs["dtype"] = self.dtype
            if is_val:
                kwargs["sample_ids"] = dataset.sample_ids

        dataset = CustomDataset(features, **kwargs)
        kwargs = {"batch_size": self.args.batch_size, "shuffle": shuffle}

        return DataLoader(dataset, **kwargs), sampled_feature_indices

    def reset_weights(self, model):
        """Reinitialize the weights of the model.

        Args:
            model (nn.Module): The model to reinitialize.
        """
        for layer in model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def reinitialize_model(self, ModelClass, model_params, boot):
        """Reinitialize a PyTorch model and optimizer with a different seed.

        This method reinitializes the model and optimizer with a different seed for each bootstrap replicate.

        Args:
            ModelClass (nn.Module): The class of the model to be reinitialized.
            model_params (dict): The parameters for the model initialization.
            boot (int): Current bootstrap replicate index.

        Returns:
            model (nn.Module): The reinitialized model.
            optimizer (torch.optim.Optimizer): The reinitialized optimizer.
            early_stop (EarlyStopping): Early stopping callback.
            lr_scheduler (LRScheduler): Learning rate scheduler.
        """
        # Clear the current model and optimizer
        torch.cuda.empty_cache()

        # Reinitialize the model
        model = ModelClass(**model_params).to(model_params["device"])

        self.reset_weights(model)

        # Reinitialize the optimizer
        optimizer = self.extract_best_params(self.best_params, model)

        early_stop, lr_scheduler = callback_init(
            optimizer, self.args, trial=None, boot=boot
        )

        return model, optimizer, early_stop, lr_scheduler

    def train_one_bootstrap(self, boot, ModelClass, train_func, train_loader):
        """Train the model with bootstrapping for a single replicate.

        This method trains the model with bootstrapping for a single replicate using the provided training function and DataLoader. The model is trained for a single bootstrap iteration and the validation, test, and prediction loaders are returned.

        Args:
            boot (int): The current bootstrap iteration.
            ModelClass (nn.Module): The class of the model to be trained.
            train_func (callable): Function to train the model.
            train_loader (DataLoader): DataLoader for the training dataset.

        Returns:
            Tuple containing the trained model, validation loader, test loader, and prediction loader.
        """
        # Generate a unique seed for this replicate
        replicate_seed = (
            self.args.seed + boot
            if self.args.seed is not None
            else np.random.choice(1e7)
        )
        thread_name = threading.current_thread().name
        self.logger.info(
            f"Bootstrap {boot} (seed={replicate_seed}) running on thread {thread_name}"
        )

        # Thread-local copies of all data
        X_pred = deepcopy(self.ds.genotypes_enc_imp[self.ds.indices["pred_indices"]])
        X_val = deepcopy(self.ds.data["X_val"])
        y_val = deepcopy(self.ds.data["y_val"])
        val_samples = deepcopy(self.ds.data["val_samples"])
        X_test = deepcopy(self.ds.data["X_test"])
        y_test = deepcopy(self.ds.data["y_test"])
        test_samples = deepcopy(self.ds.data["test_samples"])
        test_weights = np.ones(len(y_test))
        val_weights = np.ones(len(y_val))

        # Create loaders
        val_loader = DataLoader(
            CustomDataset(
                X_val,
                y_val,
                sample_weights=val_weights,
                sample_ids=val_samples,
                dtype=self.dtype,
            ),
            batch_size=self.args.batch_size,
            shuffle=False,
        )
        test_loader = DataLoader(
            CustomDataset(
                X_test,
                y_test,
                sample_weights=test_weights,
                sample_ids=test_samples,
                dtype=self.dtype,
            ),
            batch_size=self.args.batch_size,
            shuffle=False,
        )

        # Resample loaders with unique seed
        (
            train_loader,
            val_loader,
            test_loader,
            pred_loader_resamp,
        ) = self._resample_loaders(
            train_loader, val_loader, test_loader, X_pred, replicate_seed=replicate_seed
        )

        model_params = {
            "input_size": train_loader.dataset.n_features,
            "width": self.best_params["width"],
            "nlayers": self.best_params["nlayers"],
            "dropout_prop": self.best_params["dropout_prop"],
            "device": self.device,
            "output_width": train_loader.dataset.n_labels,
            "dtype": self.dtype,
        }

        model, optimizer, early_stop, lr_scheduler = self.reinitialize_model(
            ModelClass=ModelClass, model_params=model_params, boot=boot
        )

        trained_model, _ = train_func(
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

        if self.args.verbose >= 2 or self.args.debug:
            self.logger.info(f"Completed bootstrap replicate: {boot}")

        return (
            trained_model,
            val_loader,
            test_loader,
            pred_loader_resamp,
        )

    def bootstrap_training_generator(self, ModelClass, train_func, train_loader):
        """Generator to train the model with bootstrapping.

        This method trains the model with bootstrapping using the provided training function and DataLoader. The model is trained for each bootstrap iteration and the validation, test, and prediction loaders are returned.

        Args:
            ModelClass (nn.Module): The class of the model to be trained.
            train_func (callable): Function to train the model.
            train_loader (DataLoader): DataLoader for the training dataset.

        Yields:
            Tuple containing the trained model, validation loader, test loader, and prediction loader.

        Raises:
            Exception: If an unexpected error occurs during training.
        """
        n_jobs = os.cpu_count() if self.args.n_jobs == -1 else self.args.n_jobs

        if self.args.verbose >= 1:
            self.logger.info(f"Using {n_jobs} threads for bootstrapping...")

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(
                    self.train_one_bootstrap, boot, ModelClass, train_func, train_loader
                ): boot
                for boot in range(self.nboots)
            }

            for future in futures:
                boot = futures[future]
                thread_name = threading.current_thread().name

                if self.args.verbose >= 2 or self.args.debug:
                    self.logger.info(
                        f"Thread {thread_name} awaiting result for boot {boot}"
                    )
                try:
                    result = future.result()
                    self.logger.info(f"Thread {thread_name} completed boot {boot}")
                    yield result
                except Exception as exc:
                    self.logger.error(f"Bootstrap iteration {boot} exception: {exc}")
                    raise exc

    def extract_best_params(self, best_params, model):
        """Extract the best parameters from the parameter search.

        Args:
            best_params (dict): Dictionary containing the best hyperparameters.
            model (nn.Module): The model to be trained.

        Returns:
            torch.optim.Optimizer: The optimizer with the best parameters.
        """
        lr = best_params["lr"] if "lr" in best_params else best_params["learning_rate"]

        l2 = (
            best_params["l2_weight"]
            if "l2_weight" in best_params
            else best_params["l2_reg"]
        )

        # Define the criterion and optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
        return optimizer

    def save_bootstrap_results(
        self, boot, preds, write_func, dataset, test_metrics=None
    ):
        """Save the results of each bootstrap iteration.

        This method saves the results of each bootstrap iteration to file. The predictions are written to file and the test metrics are saved to a JSON file.

        Args:
            boot (int): The current bootstrap iteration.
            preds (dict): Dictionary with sampleIDs as keys and predictions made by the model in the current bootstrap iteration as values.
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

        if test_metrics is not None and dataset == "pred":
            msg = "'test_metrics' was defined for unknown predictions."
            self.logger.error(msg)
            raise TypeError(msg)

        if not isinstance(preds, dict):
            msg = f"test_preds should be a dict, but got: {type(preds)}"
            self.logger.error(msg)
            raise TypeError(msg)

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

            # If this is the first bootstrap iteration, delete existing file
            if boot_file_path.exists():
                if self.verbose >= 2:
                    self.logger.warning("Overwriting existing metrics files.")

            with open(boot_file_path, "w") as fout:
                json.dump(test_metrics, fout)

        ds = "unknown" if dataset == "pred" else dataset
        pth = outdir / "bootstrap_predictions" / ds
        pth.mkdir(exist_ok=True, parents=True)
        of = f"{self.args.prefix}_bootrep{boot}_{ds}_predictions.csv"
        outfile = pth / of

        preds_df = pd.DataFrame(preds.values(), columns=["x", "y"])
        preds_df["sampleID"] = preds.keys()

        sample_ids = preds_df["sampleID"].to_numpy()

        return write_func(preds_df, outfile, sample_ids)

    def perform_bootstrap_training(self, train_func, pred_func, write_func, ModelClass):
        """Perform bootstrapped training and prediction.

        This method performs bootstrapped training and prediction using the provided training and prediction functions. The coordinates are written to file after each bootstrap iteration.

        Args:
            train_func (callable): Function to train the model.
            pred_func (callable): Function to make predictions.
            write_func (callable): Function to write the predictions to file.
            ModelClass (nn.Module): The class of the model to be trained.

        Raises:
            Exception: If an unexpected error occurs during genotype interpolation.
            TypeError: If the model was not trained successfully and model is None.
        """
        if self.verbose >= 1:
            self.logger.info("Starting bootstrap training.")

        outdir = self.args.output_dir

        if self.args.oversample_method != "none":
            try:
                (train_loader, _, __, ___, _____) = run_genotype_interpolator(
                    self.train_loader, self.args, self.ds, self.dtype, self.plotting
                )
            except Exception as e:
                msg = f"Unexpected error occurred during genotype interpolation prior to bootstrapping: {str(e)}"
                self.logger.error(msg)
                raise e
        else:
            train_loader = self.train_loader

        bootstrap_test_preds, bootstrap_val_preds = [], []
        bootstrap_test_metrics, bootstrap_val_metrics = [], []
        bootstrap_test_sids, bootstrap_val_sids = [], []

        for boot, (
            model,
            val_loader,
            test_loader,
            pred_loader,
        ) in enumerate(
            self.bootstrap_training_generator(ModelClass, train_func, train_loader)
        ):
            if self.verbose >= 1:
                msg = f"Processing bootstrap {boot + 1}/{self.nboots}"
                self.logger.info(msg)

            outpth = Path(outdir) / "models"
            bootrep_file = outpth / f"{self.args.prefix}_model_bootrep{boot}.pt"

            if isinstance(model, tuple):
                model = model[0]

            if model is None:
                msg = f"Model was not trained successfully for bootstrap {boot}"
                self.logger.error(msg)
                raise TypeError(msg)

            val_preds, val_metrics, val_samples = pred_func(
                model,
                val_loader,
                None,
                return_truths=False,
                use_rf=self.args.use_gradient_boosting,
                bootstrap=True,
                is_val=True,
            )

            test_preds, test_metrics, test_samples = pred_func(
                model,
                test_loader,
                None,
                return_truths=False,
                use_rf=self.args.use_gradient_boosting,
                bootstrap=True,
                is_val=True,
            )

            test_preds_d = dict(zip(test_samples, test_preds))
            val_preds_d = dict(zip(val_samples, val_preds))

            bootstrap_test_preds.append(test_preds_d)
            bootstrap_val_preds.append(val_preds_d)
            bootstrap_test_metrics.append(test_metrics)
            bootstrap_val_metrics.append(val_metrics)
            bootstrap_test_sids.append(test_samples)
            bootstrap_val_sids.append(val_samples)

            self.save_bootstrap_results(
                boot,
                test_preds_d,
                write_func,
                "test",
                test_metrics=test_metrics,
            )
            self.save_bootstrap_results(
                boot,
                val_preds_d,
                write_func,
                "val",
                test_metrics=val_metrics,
            )

            torch.save(model.state_dict(), bootrep_file)

        # Process the bootstrapped predictions to generate aggregated
        # summary statistics.
        boot_test_df = self._process_boot_preds(
            outdir, bootstrap_test_preds, dataset="test"
        )
        boot_val_df = self._process_boot_preds(
            outdir, bootstrap_val_preds, dataset="val"
        )

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

    def _process_boot_preds(self, outdir, bootstrap_preds, dataset):
        """Process bootstrapped predictions to generate summary statistics.

        This method processes the bootstrapped predictions to generate summary statistics and confidence intervals for each sample group.

        Args:
            outdir (str): Output directory to use.
            bootstrap_preds (list of dict): List of dictionaries containing bootstrapped predictions.
            dataset (str): Dataset to use. Should be one of {"test", "val", "pred"}.

        Returns:
            pd.DataFrame: DataFrame containing the aggregated bootstrapped predictions.

        Raises:
            ValueError: If an invalid dataset is provided.
            TypeError: If bootstrap_preds is not a list.
        """
        if dataset not in {"val", "test", "pred"}:
            msg = f"dataset must be either 'val', 'test', or 'pred': {dataset}"
            self.logger.error(msg)
            raise ValueError(msg)

        if not isinstance(bootstrap_preds, list):
            msg = f"bootstrap_preds should be a list, but got: {type(bootstrap_preds)}"
            self.logger.error(msg)
            raise TypeError(msg)

        # Flatten bootstrap_preds and bootstrap_sids into a single list
        flat_preds = []
        flat_sids = []
        for preds in bootstrap_preds:
            for sid, coord in preds.items():
                flat_preds.append(coord)
                flat_sids.append(sid)

        # Construct the DataFrame
        bootstrap_df = pd.DataFrame(flat_preds, columns=["x", "y"])
        bootstrap_df["sampleID"] = flat_sids

        predout = self._grouped_ci_boot(bootstrap_df, dataset)

        # Save the results
        pth = Path(outdir).joinpath("bootstrap_summaries")
        pth.mkdir(exist_ok=True, parents=True)
        of = f"{self.args.prefix}_bootstrap_summary_{dataset}_predictions.csv"
        summary_outfile = pth.joinpath(of)

        predout.to_csv(summary_outfile, header=True, index=False)
        return predout

    def _grouped_ci_boot(self, df, dataset):
        """Calculate summary statistics confidence intervals for bootstrapped predictions.

        The samples are grouped by sampleID and the summary statistics and confidence intervals are calculated for each sample group consisting of all the bootstrapped replicate predictions.

        Args:
            df (pd.DataFrame): DataFrame containing bootstrapped predictions.
            dataset (str): Dataset to use. Should be one of {"test", "val", "pred"}.

        Returns:
            pd.DataFrame: DataFrame containing the mean and confidence intervals for bootstrapped predictions.

        Raises:
            ValueError: If an invalid dataset is provided.
            InvalidInputShapeError: If the input DataFrame has invalid dimensions.
        """
        df_known = None
        if self.args.known_sample_data is not None:
            fn = self.args.known_sample_data
            df_known = read_csv_with_dynamic_sep(fn)

            # Handle invalid known_sample_data file columns.
            if df_known.shape[1] != 3:
                msg = f"'--known_sample_data' file contains invalid dimensions. Expected three columns, but got {df_known.shape[1]}."
                self.logger.error(msg)
                raise InvalidInputShapeError(msg)

            col_names = ["sampleID", "x", "y"]
            df_known.columns = col_names

            for col_name, col_order, col_dtype in zip(
                col_names,
                ["first", "second", "third"],
                ["string", "numeric", "numeric"],
            ):
                self._validate_sample_data(
                    df_known, col_name, column_order=col_order, expected_dtype=col_dtype
                )

        else:
            df_known = read_csv_with_dynamic_sep(self.args.sample_data)

        if df_known is None and dataset != "pred":
            self.logger.warning("Known coordinates were not provided.")

        results = []

        n_uniq_samples = len(df["sampleID"].unique())

        if self.args.samples_to_plot is None:
            plot_indices = np.arange(n_uniq_samples)
        elif (
            isinstance(self.args.samples_to_plot, (str, int))
            and self.args.samples_to_plot.isdigit()
        ):
            plot_indices = np.random.choice(
                np.arange(n_uniq_samples),
                size=int(self.args.samples_to_plot),
                replace=False,
            )
        else:
            # Is a list of sampleIDs
            df = df.copy()
            s2p = self.args.samples_to_plot
            sids = s2p if isinstance(s2p, list) else s2p.split(",")
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

        if df_known is not None and dataset != "pred":
            dfres = dfres.sort_values(by="sampleID")
            df_known = df_known.dropna(subset=["x", "y"], how="any")
            df_known = df_known[df_known["sampleID"].isin(dfres["sampleID"])]
            dfres = dfres[dfres["sampleID"].isin(df_known["sampleID"])]

            if df_known.empty:
                self.logger.warning(
                    "No known coordinates were found for the samples in the dataset."
                )

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

    def _validate_sample_data(
        self, df, column_name, column_order="first", expected_dtype="numeric"
    ):

        dtype = check_column_dtype(df, column_name)
        if dtype != expected_dtype:
            msg = f"--known_sample_data' file is supposed to contain sampleIDs of type '{expected_dtype}' as the {column_order} column ({column_name}), but got: {dtype} data type"
            self.logger.error(msg)
            raise TypeError(msg)
