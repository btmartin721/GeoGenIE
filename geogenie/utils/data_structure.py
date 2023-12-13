import logging
import os
import sys
import warnings

os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"
warnings.filterwarnings(action="ignore", category=RuntimeWarning)


import numpy as np
import pandas as pd
import torch
from kneed import KneeLocator
from pysam import VariantFile
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import MDS, TSNE, LocallyLinearEmbedding
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from torch import float as torchfloat
from torch import tensor
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import DataLoader as GeoDataLoader

from geogenie.outliers.detect_outliers import GeoGeneticOutlierDetector
from geogenie.plotting.plotting import PlotGenIE
from geogenie.samplers.samplers import GeographicDensitySampler
from geogenie.utils.gtseq2vcf import GTseqToVCF
from geogenie.utils.utils import (
    base_to_int,
    get_iupac_dict,
    LongLatToCartesianTransformer,
    CustomDataset,
)


class DataStructure:
    """Class to hold data structure from input VCF file."""

    def __init__(self, vcf_file, verbose=False):
        """Constructor for DataStructure class.

        Args:
            genotypes (list or np.ndarray): Input genotypes loaded with pysam.
            samples (list): SampleIDs from VCF file.
        """
        self.vcf = VariantFile(vcf_file)  # pysam
        self.samples = list(self.vcf.header.samples)
        self.logger = logging.getLogger(__name__)
        self.genotypes, self.is_missing, self.genotypes_iupac = self._parse_genotypes()
        self.verbose = verbose
        self.samples_weight = None
        self.data = {}

    def define_params(self, args):
        self._params = {
            "width": args.width,
            "nlayers": args.nlayers,
            "dropout_prop": args.dropout_prop,
            "learning_rate": args.learning_rate,
            "l2_reg": args.l2_reg,
        }

    def _parse_genotypes(self):
        """Parse genotypes from the VCF file and store them in a NumPy array.

        Also, create a boolean array indicating missing data."""
        genotypes_list = []
        iupac_alleles = []
        is_missing_list = []

        for record in self.vcf:
            if self.is_biallelic(record):
                genos = []
                missing = []
                alleles = []
                for sample in record.samples.values():
                    genotype = sample.get("GT", (np.nan, np.nan))
                    genos.append(genotype)
                    missing.append(any(allele is None for allele in genotype))
                    alleles.append(sample.alleles)
                genotypes_list.append(genos)
                is_missing_list.append(missing)
                iupac_alleles.append(
                    self.map_alleles_to_iupac(alleles, get_iupac_dict())
                )
        # Convert lists to NumPy arrays for efficient computation
        genotypes_array = np.array(genotypes_list, dtype=float)
        is_missing_array = np.array(is_missing_list, dtype=bool)
        genotypes_iupac_array = np.array(iupac_alleles, dtype="object")

        return genotypes_array, is_missing_array, genotypes_iupac_array

    def map_alleles_to_iupac(self, alleles, iupac_dict):
        """
        Maps a list of allele tuples to their corresponding IUPAC nucleotide codes.

        Args:
            alleles (list of tuple of str): List of tuples representing alleles.
            iupac_dict (dict): Dictionary mapping allele tuples to IUPAC codes.

        Returns:
            list of str: List of IUPAC nucleotide codes corresponding to the alleles.
        """
        mapped_codes = []
        for allele_pair in alleles:
            if len(allele_pair) == 1:
                # Direct mapping for single alleles
                code = iupac_dict.get((allele_pair[0],), "N")
            else:
                ap = ("N", "N") if None in allele_pair else allele_pair

                # Sort the tuple for unordered pairs
                sorted_pair = tuple(sorted(ap))
                code = iupac_dict.get(sorted_pair, "N")
            mapped_codes.append(code)
        return mapped_codes

    def is_biallelic(self, record):
        """Check if number of alleles is biallelic."""
        return len(record.alleles) == 2

    def count_alleles(self):
        """Count alleles for each SNP across all samples.

        Returns:
            numpy.ndarray: 3D array of genotypes (0s and 1s) of shaep (n_loci, n_samples, 2).
        """
        if self.verbose >= 1:
            self.logger.info("Calculating allele counts.")
        allele_counts = []

        for snp_genotypes in self.genotypes:
            counts = np.zeros(2, dtype=int)  # Initialize counts for two alleles

            for genotype in snp_genotypes:
                # Ensure the genotype is an integer array, handling missing data
                int_genotype = np.array(
                    [allele if allele is not None else -1 for allele in genotype]
                )

                # Count only valid alleles (0 or 1)
                valid_alleles = int_genotype[int_genotype >= 0]
                counts += np.bincount(valid_alleles, minlength=2)

            allele_counts.append(counts)

        return np.array(allele_counts)

    def impute_missing(
        self,
        X=None,
        transform_only=False,
        strat="most_frequent",
        use_strings=False,
    ):
        """Impute missing genotypes based on allele frequency threshold.

        Args:
            X (numpy.ndarray): Data to impute. If None, then uses self.genotypes_enc instead.
            transform_only (bool): Whether to transform, but not fit.
            strat (str): Strategy to use with SimpleImputer.
            use_strings (bool): If True, uses 'N' as missing_values argument. Otherwise uses np.nan.
        """
        miss_val = "N" if use_strings else np.nan

        if not transform_only:
            self.simputer = SimpleImputer(strategy=strat, missing_values=miss_val)

        if X is None:
            X = self.genotypes_enc.copy()

        if transform_only:
            return self.simputer.transform(X)
        return self.simputer.fit_transform(X)

    def sort_samples(self, sample_data_filename):
        """Load sample_data and popmap and sort to match VCF file."""
        self.sample_data = pd.read_csv(sample_data_filename, sep="\t")

        if self.sample_data.shape[1] != 3:
            msg = f"'sample_data' must be a tab-delimited file with three columns: sampleID, x, and y. 'x' and 'y' should be longitude and latitude. However, we detected {self.sample_data.shape[1]} columns."
            self.logger.error(msg)
            raise ValueError(msg)

        self.sample_data.columns = ["sampleID", "x", "y"]
        self.sample_data["sampleID2"] = self.sample_data["sampleID"]
        self.sample_data.set_index("sampleID", inplace=True)
        self.samples = np.array(self.samples).astype(str)
        self.sample_data = self.sample_data.reindex(np.array(self.samples))

        # Sample ordering check
        self._check_sample_ordering()
        self.locs = np.array(self.sample_data[["x", "y"]])

    def normalize_locs(self):
        """Normalize locations, ignoring NaN."""
        if self.verbose >= 1:
            self.logger.info("Normalizing coordinates.")

        self.norm = LongLatToCartesianTransformer()
        self.y = self.norm.fit_transform(self.y)

        if self.verbose >= 1:
            self.logger.info("Done normalizing.")

    def _check_sample_ordering(self):
        """Validate sample ordering between 'sample_data' and VCF files."""
        for i, x in enumerate(self.samples):
            if self.sample_data["sampleID2"].iloc[i] != x:
                msg = "Invalid sample ordering. sample IDs in 'sample_data' must match the order in the VCF file."
                self.logger.error(msg)
                raise ValueError(msg)

    def encode_sequence(self, seq):
        """Encode DNA sequence as integers.

        4 corresponds to default value for "N".

        Args:
            seq (str): Whole sequence, not delimited.

        Returns:
            list: Integer-encoded IUPAC genotypes.
        """
        return [base_to_int().get(base, 4) for base in seq]

    def snps_to_012(self, min_mac=2, max_snps=None, return_values=True):
        """Convert IUPAC SNPs to 012 encodings."""
        if self.verbose >= 1:
            self.logger.info("Converting SNPs to 012-encodings.")

        self.genotypes_enc = np.sum(
            self.genotypes, axis=-1, where=~np.all(np.isnan(self.genotypes))
        )

        allele_counts = np.apply_along_axis(
            lambda x: np.bincount(x[~np.isnan(x)].astype(int), minlength=3),
            axis=1,
            arr=self.genotypes_enc,
        )

        self.genotypes_enc = self.filter_gt(
            self.genotypes_enc, min_mac, max_snps, allele_counts
        )

        if self.verbose >= 1:
            self.logger.info("Input SNP data converted to 012-encodings.")

        if return_values:
            return self.genotypes_enc.T
        else:
            self.genotypes_enc = self.genotypes_enc.T

    def filter_gt(self, gt, min_mac, max_snps, allele_counts):
        """Filter genotypes based on minor allele count and random subsets (max_snps)."""
        if min_mac > 1:
            mac = 2 * allele_counts[:, 2] + allele_counts[:, 1]
            gt = gt[mac >= min_mac, :]

        if max_snps is not None:
            gt = gt[
                np.random.choice(range(gt.shape[0]), max_snps, replace=False),
                :,
            ]

        return gt

    def snps_to_int(self, min_mac=2, max_snps=None, return_values=True):
        """Convert SNPs to integer encodings 0-14 (including IUPAC)."""
        snps = self.genotypes_iupac.tolist()
        snps = ["".join(x) for x in snps]
        snps = list(map(self.encode_sequence, snps))
        snps_enc = np.array(snps, dtype="int8")

        gt_012 = np.sum(self.genotypes, axis=-1).astype("int8")

        allele_counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=3),
            axis=1,
            arr=gt_012,
        )

        self.genotypes_enc = self.filter_gt(
            snps_enc,
            min_mac,
            max_snps,
            allele_counts,
        )

        if return_values:
            return self.genotypes_enc

    def split_train_test(self, train_split, val_split, seed):
        """
        Split data into training, validation, and testing sets, handling NaN values in locations.

        Args:
            train_split (float): Proportion of the data to be used for training.
            val_split (float): Proportion of the training data to be used for validation.
            seed (int): Random seed for reproducibility.

        Returns:
            tuple: tuple containing indices and data for training, validation, and testing.
        """

        if self.verbose >= 1:
            self.logger.info(
                "Splitting data into train, validation, and test datasets.",
            )

        if train_split - val_split <= 0:
            raise ValueError(
                "The difference between train_split and val_split must be >= 0."
            )

        # Split non-NaN samples into training + validation and test sets
        (
            X_train_val,
            X_test,
            y_train_val,
            y_test,
            train_val_indices,
            test_indices,
        ) = train_test_split(
            self.X,
            self.y,
            self.model_indices,
            train_size=train_split,
            random_state=seed,
        )

        # Split training data into actual training and validation sets
        X_train, X_val, y_train, y_val, train_indices, val_indices = train_test_split(
            X_train_val,
            y_train_val,
            train_val_indices,
            test_size=val_split,
            random_state=seed,
        )

        data = {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "X_pred": self.X_pred,
            "X": self.X,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "y": self.y,
        }
        self.data.update(data)

        self.indices = {
            "train_val_indices": train_val_indices,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": test_indices,
            "pred_indices": self.pred_indices,
            "true_indices": self.true_indices,
        }

        if self.verbose >= 1:
            self.logger.info("Created train, validation, and test datasets.")

    def map_outliers_through_filters(self, original_indices, filter_stages, outliers):
        """
        Maps outlier indices through multiple filtering stages back to the original dataset.

        Args:
            original_indices (np.array): Array of original indices before any filtering.
            filter_stages (list of np.array): List of arrays of indices after each filtering stage.
            outliers (np.array): Outlier indices in the most filtered dataset.

        Returns:
            np.array: Mapped outlier indices in the original dataset.
        """
        current_indices = outliers
        for stage_indices in reversed(filter_stages):
            current_indices = stage_indices[current_indices]
        return original_indices[current_indices]

    def load_and_preprocess_data(self, args):
        """Wrapper method to load and preprocess data."""

        self.plotting = PlotGenIE(
            "cpu",
            args.output_dir,
            args.prefix,
            show_plots=args.show_plots,
            fontsize=args.fontsize,
        )

        if self.verbose >= 1:
            self.logger.info("Loading and preprocessing input data...")

        # Load and sort sample data
        self.sort_samples(args.sample_data)

        if args.model_type in ["mlp", "gcn"]:
            self.snps_to_012(
                min_mac=args.min_mac,
                max_snps=args.max_SNPs,
                return_values=False,
            )
        elif args.model_type == "transformer":
            self.snps_to_int(
                min_mac=args.min_mac,
                max_snps=args.max_SNPs,
                return_values=False,
            )

        if self.genotypes_enc.shape[0] != self.locs.shape[0]:
            msg = f"Invalid input shapes for genotypes and coorindates. The number of rows (samples) must be equal, but got: {self.genotypes_enc.shape}, {self.locs.shape}"
            self.logger.error(msg)
            raise ValueError(msg)

        # Impute missing data and embed.
        X = self.impute_missing(self.genotypes_enc)

        X = self.embed(
            args,
            X=X,
            alg=args.embedding_type,
            full_dataset_only=True,
            transform_only=False,
        )

        indices = np.arange(self.locs.shape[0])
        self.pred_mask = ~np.isnan(self.locs).any(axis=1)
        X = X[self.pred_mask, :]
        y = self.locs[self.pred_mask, :]
        index = self.samples[self.pred_mask]

        # Define true_indices and pred_indices
        self.all_indices = indices
        self.true_indices = indices[self.pred_mask]  # True if is not nan.
        self.pred_indices = indices[~self.pred_mask]  # True if is nan.

        # Store indices after filtering for nan
        filter_stage_indices = [self.true_indices]

        if args.verbose >= 1:
            self.logger.info(
                f"Found {np.sum(self.pred_mask)} known samples and {np.sum(~self.pred_mask)} samples to predict."
            )

        if X.shape[0] != y.shape[0]:
            msg = f"Invalid input shapes for X and y. The number of rows (samples) must be equal, but got: {X.shape}, {y.shape}"
            self.logger.error(msg)
            raise ValueError(msg)

        # Assuming GeoGeneticOutlierDetector is implemented elsewhere
        outlier_detector = GeoGeneticOutlierDetector(
            pd.DataFrame(X, index=index),
            pd.DataFrame(y, index=index),
            output_dir=args.output_dir,
            prefix=args.prefix,
            n_jobs=args.n_jobs,
            url=args.shapefile_url,
            buffer=0.1,
            show_plots=args.show_plots,
            seed=args.seed,
            debug=True,
            verbose=args.verbose,
        )

        outliers = outlier_detector.composite_outlier_detection(
            sig_level=args.significance_level,
            maxk=args.maxk,
            min_nn_dist=args.min_nn_dist,
        )

        all_outliers = np.concatenate((outliers["geographic"], outliers["genetic"]))

        # Returns outiler indieces, remapped.
        mapped_all_outliers = self.map_outliers_through_filters(
            indices, filter_stage_indices, all_outliers
        )

        # Remove mapped outliers from your data
        mask = np.ones(len(self.all_indices), dtype=bool)
        mask[np.isin(self.all_indices, mapped_all_outliers)] = False
        mask[np.isin(self.all_indices, self.pred_indices)] = False
        pred_mask = np.zeros(len(self.all_indices), dtype=bool)
        pred_mask[np.isin(self.all_indices, self.pred_indices)] = True

        # Apply mask to true_indices and respective datasets
        self.X = self.genotypes_enc[mask, :]
        self.y = self.locs[mask, :]
        self.X_pred = self.genotypes_enc[pred_mask, :]
        self.model_indices = self.all_indices[mask]

        self.logger.info(
            f"{self.X.shape[0]} samples remaining after removing {len(all_outliers)} outliers."
        )

        # Continue with the rest of your pipeline
        self.evaluate_outliers(
            index, outliers["geographic"], outliers["genetic"], "Composite"
        )
        self.normalize_locs()
        self.split_train_test(args.train_split, args.val_split, args.seed)

        for k, v in self.data.items():
            if k.startswith("X"):
                tonly = False if k == "X_train" else True

                # Impute missing values, eliminate data leakage.
                self.data[k] = self.impute_missing(X=v, transform_only=tonly)

                self.data[k] = self.embed(
                    args,
                    X=self.data[k],
                    alg=args.embedding_type,
                    transform_only=tonly,
                )

        if args.verbose >= 1:
            self.logger.info("Data split into train, val, and test sets.")
            self.logger.info("Creating DataLoader objects.")

        # Creating DataLoaders
        self.train_loader = self.create_dataloaders(
            self.data["X_train"],
            self.data["y_train"],
            args.class_weights,
            args.batch_size,
            args,
            model_type=args.model_type,
        )

        # Creating DataLoaders
        self.val_loader = self.create_dataloaders(
            self.data["X_val"],
            self.data["y_val"],
            False,
            args.batch_size,
            args,
            model_type=args.model_type,
        )

        # Creating DataLoaders
        self.test_loader = self.create_dataloaders(
            self.data["X_test"],
            self.data["y_test"],
            False,
            args.batch_size,
            args,
            model_type=args.model_type,
        )

        if args.model_type == "gcn":
            self.train_loader, self.train_dataset = self.train_loader
            self.val_loader, self.val_dataset = self.val_loader
            self.test_loader, self.test_dataset = self.test_loader

        if args.verbose >= 1:
            self.logger.info("DataLoaders created succesfully.")

        if args.verbose >= 1:
            self.logger.info("Data loading and preprocessing completed.")

    def evaluate_outliers(
        self, train_samples, outlier_geo_indices, outlier_gen_indices, dt
    ):
        if not isinstance(train_samples, pd.Series):
            train_samples = pd.Series(train_samples, name="ID")

        """Evaluate and plot results from GeoGenOutlierDetector."""
        y_pred = self.process_pred(
            train_samples, outlier_geo_indices, outlier_gen_indices, dt
        )
        y_true = self.process_truths(train_samples, dt)
        self.plotting.plot_confusion_matrix(y_true["Label"], y_pred["Label"], dt)

        self.logger.info(
            f"Accuracy: {accuracy_score(y_true['Label'], y_pred['Label'])}"
        )
        self.logger.info(f"F1 Score: {f1_score(y_true['Label'], y_pred['Label'])}")

    def process_truths(self, train_samples, label_type):
        """Process true values."""
        truths = pd.read_csv("data/real_outliers.csv", sep=" ")
        truths["method"] = truths["method"].str.replace('"', "")
        truths["method"] = truths["method"].str.replace("'", "")
        y_true = pd.DataFrame(columns=["ID"])
        y_true["ID"] = train_samples
        y_true["Label"] = y_true["ID"].isin(truths["ID"]).astype(int)
        y_true["Type"] = "True"
        return y_true.sort_values(by=["ID"])

    def process_pred(
        self, train_samples, outlier_geo_indices, outlier_gen_indices, label_type
    ):
        """Process predictions."""
        # Create the y_pred DataFrame
        y_pred = pd.DataFrame(columns=["ID"])
        y_pred["ID"] = train_samples
        y_pred[label_type] = 0
        y_pred["Type"] = "Pred"

        # Convert outlier indices to IDs
        outlier_geo_ids = train_samples.iloc[outlier_geo_indices]
        outlier_gen_ids = train_samples.iloc[outlier_gen_indices]

        outlier_ids = pd.concat([outlier_geo_ids, outlier_gen_ids])

        # Mark the outliers in y_pred based on IDs
        y_pred["Label"] = y_pred["ID"].isin(outlier_ids).astype(int)
        return y_pred.sort_values(by=["ID"])

    @np.errstate(all="warn")
    def embed(
        self,
        args,
        X=None,
        alg="polynomial",
        full_dataset_only=False,
        transform_only=False,
    ):
        """Embed SNP data using one of several dimensionality reduction techniques.

        Args:
            args (argparse.Namespace): User-supplied arguments.
            X (numpy.ndarray): Data to embed. If is None, then self.genotypes_enc gets used instead.
            alg (str): Algorithm to use. Valid arguments include: 'polynomial', 'pca', 'tsne', 'polynomial', 'mds', and 'none' (no embedding). Defaults to 'polynomial'.
            return_values (bool): If True, returns the embddings. If False, sets embedding as class attributes, self.data and self.indices. Defaults to False.
            full_dataset_only (bool): If True, only embed and return full dataset.

        Warnings:
            Setting 'polynomial_degree' > 2 can lead to extremely large computational overhead. Do so at your own risk!!!

        Returns:
            tuple(numpy.ndarray): train, test, validation, prediction, and full (unsplit) datasets. Only returns of return_values is True.

        ToDo:
            Make T-SNE plot.
            Make MDS plot.
        """
        if X is None:
            X = self.genotypes_enc.copy()

        if args.model_type != "transformer":
            do_embed = True
        else:
            do_embed = False
        if alg.lower() == "polynomial":
            emb = PolynomialFeatures(args.polynomial_degree)
        elif alg.lower() == "pca":
            n_components = args.n_components
            if args.n_components is None and not transform_only:
                n_components = self.get_num_pca_comp(X)
                emb = PCA(n_components=n_components, random_state=args.seed)

                if n_components is None:
                    self.logger.error(
                        "n_componenets must be defined, but got NoneType."
                    )
                    raise TypeError("n_componenets must be defined, but got NoneType.")
        elif alg.lower() == "tsne":
            # TODO: Make T-SNE plot.
            emb = TSNE(
                n_components=args.n_components,
                perplexity=args.perplexity,
                random_state=args.seed,
            )
        elif alg.lower() == "mds":
            # TODO: Make MDS plot.
            emb = MDS(
                n_components=args.n_components,
                n_init=args.n_init,
                n_jobs=args.n_jobs,
                random_state=args.seed,
                normalized_stress="auto",
            )
        elif alg.lower() == "lle":

            class LocallyLinearEmbeddingWrapper(LocallyLinearEmbedding):
                def predict(self, X):
                    return self.transform(X)

            def lle_reconstruction_scorer(estimator, X, y=None):
                """
                Compute the negative reconstruction error for an LLE model to use as a scorer.
                GridSearchCV assumes that higher score values are better, so the reconstruction
                error is negated.

                Args:
                    estimator (LocallyLinearEmbedding): Fitted LLE model.
                    X (numpy.ndarray): Original high-dimensional data.

                Returns:
                    float: Negative reconstruction error.
                """
                return -estimator.reconstruction_error_

            # Non-linear dimensionality reduction.
            # default max_iter often doesn't converge.
            emb = LocallyLinearEmbeddingWrapper(
                method="modified", max_iter=1000, random_state=args.seed
            )

            param_grid = {
                "n_neighbors": np.linspace(
                    5, args.maxk, num=10, endpoint=True, dtype=int
                ),
                "n_components": np.linspace(
                    2, args.maxk, num=10, endpoint=True, dtype=int
                ),
            }

            grid = GridSearchCV(
                emb,
                param_grid=param_grid,
                cv=5,
                n_jobs=args.n_jobs,
                verbose=0,
                scoring=lle_reconstruction_scorer,
                refit=True,
                error_score=-np.inf,
            )

        elif alg.lower() == "none":
            # Use raw encoded data.
            do_embed = False
        else:
            raise ValueError(f"Invalid 'alg' value pasesed to 'embed()': {alg}")

        if not transform_only and do_embed:
            self.ssc = StandardScaler()
            X = self.ssc.fit_transform(X)
            if alg.lower() == "lle":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    grid.fit(X)
                self.emb = grid.best_estimator_
                if args.verbose >= 1:
                    self.logger.info(
                        f"Best parameters for LocallyLinearEmbedding: {grid.best_params_}"
                    )
                    self.logger.info(
                        f"Best score for LocallyLinearEmbedding: {grid.best_score_}"
                    )
            else:
                self.emb = emb

        if do_embed:
            if full_dataset_only:
                X = self.ssc.fit_transform(X)
                X = self.emb.fit_transform(X)
                return X

            if not transform_only:
                X = self.ssc.fit_transform(X)
                X = self.emb.fit_transform(X)
                return X

            X = self.ssc.transform(X)
            X = self.emb.transform(X)
            return X
        else:
            return X.copy()

    def get_num_pca_comp(self, x):
        """Get optimal number of PCA components.

        Args:
            x (numpy.ndarray): Dataset to fit PCA to.

        Returns:
            int: Optimal number of principal components to use.
        """
        pca = PCA().fit(x)

        vr = np.cumsum(pca.explained_variance_ratio_)

        x = range(1, len(vr) + 1)
        kneedle = KneeLocator(
            x,
            vr,
            S=1.0,
            curve="concave",
            direction="increasing",
        )

        knee = int(np.ceil(kneedle.knee))
        self.plotting.plot_pca_curve(x, vr, knee)
        return knee

    def create_dataloaders(
        self,
        X,
        y,
        class_weights,
        batch_size,
        args,
        model_type="mlp",
    ):
        """
        Create dataloaders for training, testing, and validation datasets.

        Args:
            X (numpy.ndarray or list of PyG Data objects): X dataset. Train, test, or validation.
            y (numpy.ndarray or None): Target data (train, test, or validation). None for GNN.
            class_weights (bool): If True, calculates class weights for weighted sampling.
            batch_size (int): Batch size to use with model.
            args (argparse.Namespace): User-supplied arguments.
            model_type (str): Type of the model: 'gcn', 'transformer', or 'mlp'.

        Returns:
            torch.utils.data.DataLoader: DataLoader object suitable for the specified model type.
        """
        # For GNNs
        if model_type == "gcn":
            if not isinstance(X, list):
                try:
                    X = X.tolist()
                except Exception:
                    raise ValueError(
                        "For GNNs, X should be a list of PyG Data objects",
                    )
            # Convert data to TensorDatasets
            dataset = TensorDataset(
                tensor(X, dtype=torchfloat), tensor(y, dtype=torchfloat)
            )

            return GeoDataLoader(X, batch_size=batch_size, shuffle=True), dataset

        # For MLPs and Transformers
        else:
            # Custom sampler - density-based.
            weighted_sampler = self.get_sample_weights(
                self.norm.inverse_transform(y), class_weights, args
            )

            dataset = CustomDataset(X, y, sample_weights=self.samples_weight)

            # Create DataLoaders
            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=weighted_sampler,
            )

    def get_sample_weights(self, y, class_weights, args):
        """Gets inverse sample_weights based on sampling density.

        Only performed if 'class_weights' is True.

        Uses scikit-learn KernelDensity with a grid search to estimate the optimal bandwidth for GeographicDensitySampler.

        Args:
            y (numpy.ndarray): Target values.
            class_weights (bool): Whether to use weights.
            args (argparse.Namespace): User-supplied arguments.
        """
        # Initialize weighted sampler if class weights are used
        weighted_sampler = None
        if class_weights:
            if args.verbose >= 1:
                self.logger.info("Estimating sampling density weights...")
                self.logger.info(
                    "Searching for optimal kernel density bandwidth for geographic density sampler..."
                )

            # Weight by sampling density.
            weighted_sampler = GeographicDensitySampler(
                y,
                focus_regions=args.focus_regions,
                use_kmeans=args.use_kmeans,
                use_kde=args.use_kde,
                w_power=args.w_power,
                max_clusters=args.max_clusters,
                max_neighbors=args.max_neighbors,
            )

            if args.verbose >= 1:
                self.logger.info("Done estimating sample weights.")

            self.samples_weight = weighted_sampler.weights
            self.samples_weight_indices = weighted_sampler.indices
        return weighted_sampler

    def read_gtseq(
        self,
        filename,
        output_dir,
        prefix,
        loci2drop=None,
        subset_vcf_gtseq=None,
    ):
        if self.verbose >= 1:
            self.logger.info("Reading GTSeq input file.")
        # Create an instance of the class with the path to your data file
        converter = GTseqToVCF(filename, str2drop=loci2drop)

        # Load and preprocess the data
        converter.load_data()
        converter.parse_sample_column()
        converter.calculate_ref_alt_alleles()
        converter.transpose_data()

        # Generate the VCF content
        converter.create_vcf_header()
        converter.format_genotype_data()

        output_dir = os.path.join(output_dir, f"{prefix}_Genotypes.vcf")

        # Write the VCF to a file
        converter.write_vcf(output_dir)

        if subset_vcf_gtseq is not None:
            # Subset the VCF by locus IDs and write to a new file
            converter.subset_vcf_by_locus_ids(
                subset_vcf_gtseq, output_dir, f"{prefix}_gtseq_loci_subset.vcf"
            )

        if self.verbose >= 1:
            self.logger.info("Successfully loaded GTSeq data.")

    @property
    def params(self):
        """Getter for the params dictionary.

        Returns:
            dict: Parameters dictionary.
        """
        return self._params

    @params.setter
    def params(self, value):
        """Update the params dictionary.

        Args:
            value (dict): Dictionary to update params with.
        """
        self._params.update(value)
