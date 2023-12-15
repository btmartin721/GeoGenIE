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
from geogenie.utils.scorers import LocallyLinearEmbeddingWrapper
from geogenie.utils.utils import (
    CustomDataset,
    LongLatToCartesianTransformer,
    get_iupac_dict,
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

        self.mask = np.ones_like(self.samples, dtype=bool)
        self.simputer = SimpleImputer(strategy="most_frequent", missing_values=np.nan)

    def define_params(self, args):
        self._params = vars(args)

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

    def impute_missing(self, X, transform_only=False):
        """Impute missing genotypes based on allele frequency threshold.

        Args:
            X (numpy.ndarray): Data to impute.
            transform_only (bool): Whether to transform, but not fit.
            strat (str): Strategy to use with SimpleImputer.
            use_strings (bool): If True, uses 'N' as missing_values argument. Otherwise uses np.nan.
        """
        if transform_only:
            return self.simputer.transform(X)
        self.simputer = clone(self.simputer)
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

    def normalize_target(self, placeholder=False):
        """Normalize locations, ignoring NaN."""
        if self.verbose >= 1:
            self.logger.info("Normalizing coordinates.")

        # Turn off placeholder to actually do transformation.
        self.norm = LongLatToCartesianTransformer(placeholder=placeholder)
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
                "The difference between train_split - val_split must be >= 0."
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
            self.true_idx,
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

        if args.model_type.lower() == "mlp":
            self.snps_to_012(
                min_mac=args.min_mac,
                max_snps=args.max_SNPs,
                return_values=False,
            )
        else:
            raise NotImplementedError(
                f"Only mlp model is currently implemented, but got: {args.model_type}"
            )

        # Make sure features and target have same number of rows.
        self.validate_feature_target_len()

        # Impute missing data and embed.
        X = self.impute_missing(self.genotypes_enc)

        # Do embedding (e.g., PCA, LLE)
        X = self.embed(
            args,
            X=X,
            alg=args.embedding_type,
            full_dataset_only=True,
            transform_only=False,
        )

        # Define true_indices and pred_indices
        self.pred_mask = ~np.isnan(self.locs).any(axis=1)

        X, indices, y, index = self.setup_index_masks(X)
        self.all_indices = indices
        self.true_indices = indices[self.pred_mask]  # True if is not nan.
        self.pred_indices = indices[~self.pred_mask]  # True if is nan.

        filter_stage_indices = [self.true_indices]

        if args.verbose >= 1:
            self.logger.info(
                f"Found {np.sum(self.pred_mask)} known samples and {np.sum(~self.pred_mask)} samples to predict."
            )

        if X.shape[0] != y.shape[0]:
            msg = f"Invalid input shapes for X and y. The number of rows (samples) must be equal, but got: {X.shape}, {y.shape}"
            self.logger.error(msg)
            raise ValueError(msg)

        if args.detect_outliers:
            all_outliers = self.run_outlier_detection(
                args, X, indices, y, index, filter_stage_indices
            )

        # Here X has not been imputed.
        self.X, self.y, self.X_pred, self.true_idx = self.extract_datasets()

        if args.detect_outliers:
            self.logger.info(
                f"{self.X.shape[0]} samples remaining after removing {len(all_outliers)} outliers."
            )

        # placeholder=True makes it not do transform.
        self.normalize_target(placeholder=True)
        self.split_train_test(args.train_split, args.val_split, args.seed)

        for k, v in self.data.items():
            if k.startswith("X"):
                tonly = False if k == "X_train" else True

                # Impute missing values, eliminate data leakage.
                imputed = self.impute_missing(v, transform_only=tonly)

                self.data[k] = self.embed(
                    args,
                    X=imputed,
                    alg=args.embedding_type,
                    transform_only=tonly,
                )

        if args.verbose >= 1:
            self.logger.info("Data split into train, val, and test sets.")
            self.logger.info("Creating DataLoader objects.")

        # Creating DataLoaders
        self.train_loader = self.call_create_dataloaders(
            self.data["X_train"], self.data["y_train"], args, False
        )
        self.val_loader = self.call_create_dataloaders(
            self.data["X_val"], self.data["y_val"], args, True
        )
        self.test_loader = self.call_create_dataloaders(
            self.data["X_test"], self.data["y_test"], args, True
        )

        if args.verbose >= 1:
            self.logger.info("DataLoaders created succesfully.")
            self.logger.info("Data loading and preprocessing completed.")

    def extract_datasets(self):
        self.mask[np.isin(self.all_indices, self.pred_indices)] = False
        pred_mask = np.zeros(len(self.all_indices), dtype=bool)
        pred_mask[np.isin(self.all_indices, self.pred_indices)] = True

        return (
            self.genotypes_enc[self.mask, :],
            self.locs[self.mask, :],
            self.genotypes_enc[pred_mask, :],
            self.all_indices[self.mask],
        )

    def validate_feature_target_len(self):
        if self.genotypes_enc.shape[0] != self.locs.shape[0]:
            msg = f"Invalid input shapes for genotypes and coorindates. The number of rows (samples) must be equal, but got: {self.genotypes_enc.shape}, {self.locs.shape}"
            self.logger.error(msg)
            raise ValueError(msg)

    def setup_index_masks(self, X):
        indices = np.arange(self.locs.shape[0])
        X = X[self.pred_mask, :]
        y = self.locs[self.pred_mask, :]
        index = self.samples[self.pred_mask]

        # Store indices after filtering for nan
        return X, indices, y, index

    def run_outlier_detection(self, args, X, indices, y, index, filter_stage_indices):
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
            debug=False,
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
        self.mask[np.isin(self.all_indices, mapped_all_outliers)] = False

        return all_outliers

    def call_create_dataloaders(self, X, y, args, is_val):
        return self.create_dataloaders(
            X,
            y,
            args.batch_size,
            args,
            is_val=is_val,
        )

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
                scoring=LocallyLinearEmbeddingWrapper.lle_reconstruction_scorer,
                refit=True,
                error_score=-np.inf,  # metric is maximized.
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
        batch_size,
        args,
        is_val=False,
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
        # Custom sampler - density-based.
        weighted_sampler = self.get_sample_weights(self.norm.inverse_transform(y), args)

        if self.samples_weight is None:
            self.samples_weight = torch.ones(y.shape[0], dtype=torch.float32)

        dataset = CustomDataset(X, y, sample_weights=self.samples_weight)

        # Create DataLoader
        kwargs = {"batch_size": batch_size}
        if args.use_weighted in ["sampler", "both"]:
            kwargs["sampler"] = weighted_sampler
        else:
            kwargs["shuffle"] = False if is_val else True
        return DataLoader(dataset, **kwargs)

    def get_sample_weights(self, y, args):
        """Gets inverse sample_weights based on sampling density.

        Only performed if 'class_weights' is True.

        Uses scikit-learn KernelDensity with a grid search to estimate the optimal bandwidth for GeographicDensitySampler.

        Args:
            y (numpy.ndarray): Target values.
            class_weights (bool): Whether to use weights.
            args (argparse.Namespace): User-supplied arguments.
        """
        # Initialize weighted sampler if class weights are used
        self.weighted_sampler = None
        if args.use_weighted in ["sampler", "loss", "both"]:
            if args.verbose >= 1:
                self.logger.info("Estimating sampling density weights...")
                self.logger.info(
                    "Searching for optimal kernel density bandwidth for geographic density sampler..."
                )

            # Weight by sampling density.
            weighted_sampler = GeographicDensitySampler(
                pd.DataFrame(y, columns=["x", "y"]),
                focus_regions=args.focus_regions,
                use_kmeans=args.use_kmeans,
                use_kde=args.use_kde,
                w_power=args.w_power,
                max_clusters=args.max_clusters,
                max_neighbors=args.max_neighbors,
            )

            if args.verbose >= 1:
                self.logger.info("Done estimating sample weights.")

            self.weighted_sampler = weighted_sampler
            self.samples_weight = weighted_sampler.weights
            self.samples_weight_indices = weighted_sampler.indices
        return self.weighted_sampler

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
