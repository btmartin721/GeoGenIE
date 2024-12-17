import logging
import os
import warnings
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"
warnings.filterwarnings(action="ignore", category=RuntimeWarning)


import numpy as np
import pandas as pd
import pysam
import torch
from kneed import KneeLocator
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA, KernelPCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import PolynomialFeatures
from torch.utils.data import DataLoader

from geogenie.outliers.detect_outliers import GeoGeneticOutlierDetector
from geogenie.plotting.plotting import PlotGenIE
from geogenie.samplers.samplers import GeographicDensitySampler
from geogenie.utils.data import CustomDataset
from geogenie.utils.exceptions import (EmbeddingError, InvalidInputShapeError,
                                       InvalidSampleDataError,
                                       OutlierDetectionError,
                                       SampleOrderingError)
from geogenie.utils.scorers import LocallyLinearEmbeddingWrapper
from geogenie.utils.transformers import MCA, MinMaxScalerGeo
from geogenie.utils.utils import get_iupac_dict, read_csv_with_dynamic_sep


class DataStructure:
    """Class to hold data structure from input VCF file.

    High level class overview. Key class functionalities include data loading, preprocessing, transformation, and machine learning model preparation.

    Initialization and Data Parsing - It loads VCF (Variant Call Format) files using pysam, processes genotypes, and handles missing data.

    Data Transformation and Imputation - Implements methods for allele counting, imputing missing genotypes, normalizing data, and transforming genotypes to various encodings.

    Data Splitting - Facilitates splitting data into training, validation, and test sets.

    Outlier Detection - Includes methods for detecting and handling outliers in the data.

    Data Preprocessing and Embedding: The class contains methods for preprocessing data, including scaling, dimensionality reduction, and embedding using various techniques like PCA, t-SNE, MCA, etc.

    Data Analysis and Visualization - The script integrates with the geogenie library for tasks like outlier detection and plotting.

    Machine Learning and Data Loading - It includes functionalities for creating data loaders (using torch) and preparing datasets for machine learning tasks, with support for different data sampling strategies and weightings.

    Utility Methods - The script provides additional utility methods for tasks like reading GTseq data, setting parameters, and selecting optimal components for different data transformations.

    In summary, the script is designed for comprehensive genomic data analysis, offering capabilities for data loading, preprocessing, transformation, machine learning model preparation, and visualization. It is structured to handle data from VCF files, process it through various analytical and transformational steps, and prepare it for further analysis or machine learning tasks.
    """

    def __init__(self, vcf_file, verbose=False, dtype=torch.float32, debug=False):
        """Constructor for DataStructure class.

        Args:
            vcf_file (str): VCF filename to load.
            verbose (bool): Whether to enable verbosity. Default is False.
            dtype (torch.dtype): Data type for tensors. Default is torch.float32.
            debug (bool): Whether to enable debug mode. Default is False.
        """
        self.vcf_file = vcf_file
        self.verbose = verbose
        self.dtype = dtype
        self.debug = debug
        self.data = {}
        self.logger = logging.getLogger(__name__)

        # Attempt to load the VCF file with a fallback on decompression and
        # recompression
        self.vcf = self._load_vcf_file()

        # Continue with initialization after loading VCF
        self.samples = list(self.vcf.header.samples)
        self.genotypes, self.is_missing, self.genotypes_iupac = self._parse_genotypes()
        self.mask = np.ones_like(self.samples, dtype=bool)
        self.simputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        self.norm = MinMaxScalerGeo()

    def _load_vcf_file(self):
        """Tries to load the VCF file and handles decompression and recompression if necessary.

        Returns:
            pysam.VariantFile: Loaded VCF file.
        """
        try:
            return pysam.VariantFile(self.vcf_file)
        except NotImplementedError as e:
            self.logger.warning(f"Encountered issue with VCF compression: {e}")
            return self._decompress_recompress_index_and_load_vcf()

    def _decompress_recompress_index_and_load_vcf(self):
        """Decompresses, recompresses, indexes, and reloads the VCF file.

        Returns:
            pysam.VariantFile: Recompressed and reloaded VCF file.
        """
        decompressed_vcf = self._decompress_vcf()
        recompressed_vcf = self._recompress_vcf(decompressed_vcf)
        self._index_vcf(recompressed_vcf)
        self._cleanup_decompressed_file(decompressed_vcf)
        return pysam.VariantFile(recompressed_vcf)

    def _decompress_vcf(self):
        """Decompresses the gzipped VCF file and returns the decompressed file path.

        Returns:
            str: Path to the decompressed VCF file.
        """
        decompressed_vcf_file = self.vcf_file.replace(".gz", "")
        with (
            pysam.BGZFile(self.vcf_file, "rb") as compressed_vcf,
            open(decompressed_vcf_file, "wb") as decompressed_vcf,
        ):
            decompressed_vcf.write(compressed_vcf.read())
        self.logger.info(f"Decompressed VCF file: {decompressed_vcf_file}")
        return decompressed_vcf_file

    def _recompress_vcf(self, decompressed_vcf_file):
        """Recompresses the VCF file using bgzip and returns the recompressed file path.

        Args:
            decompressed_vcf_file (str): Path to the decompressed VCF file.

        Returns:
            str: Path to the recompressed VCF file.
        """
        dvcf = Path(decompressed_vcf_file)
        recompressed_vcf_file = (
            str(dvcf.with_name(dvcf.stem + "_recompressed.vcf")) + ".gz"
        )
        with (
            open(decompressed_vcf_file, "rb") as decompressed_vcf,
            pysam.BGZFile(recompressed_vcf_file, "wb") as recompressed_vcf,
        ):
            recompressed_vcf.write(decompressed_vcf.read())
        self.logger.info(f"Recompressed VCF file: {recompressed_vcf_file}")
        return recompressed_vcf_file

    def _index_vcf(self, recompressed_vcf_file):
        """Indexes the recompressed VCF file using tabix.

        Args:
            recompressed_vcf_file (str): Path to the recompressed VCF file.
        """
        pysam.tabix_index(recompressed_vcf_file, preset="vcf", force=True)
        self.logger.info(f"Indexed VCF file: {recompressed_vcf_file}")

    def _cleanup_decompressed_file(self, decompressed_vcf_file):
        """Deletes the decompressed VCF file to save space.

        Args:
            decompressed_vcf_file (str): Path to the decompressed VCF file.
        """
        Path(decompressed_vcf_file).unlink()
        self.logger.info(f"Deleted decompressed VCF file: {decompressed_vcf_file}")

    def _parse_genotypes(self):
        """Parse genotypes from the VCF file and store them in a NumPy array.

        Also, create a boolean array indicating missing data.

        Returns:
            tuple: A tuple containing genotypes array, missing data array, and IUPAC encoded genotypes array.
        """
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
        """Maps a list of allele tuples to their corresponding IUPAC nucleotide codes.

        Args:
            alleles (List[tuple]): List of tuples representing alleles.
            iupac_dict (dict): Dictionary mapping allele tuples to IUPAC codes.

        Returns:
            List[str]: List of IUPAC nucleotide codes corresponding to the alleles.
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
        """Check if number of alleles is biallelic.

        Args:
            record (pysam.VariantRecord): A VCF record.

        Returns:
            bool: True if the record is biallelic, False otherwise.
        """
        return len(record.alleles) == 2

    def define_params(self, args):
        """Defines and updates class attrubutes from an argparse.Namespace object.

        This method sets the class attributes based on the parameters provided in the argparse.Namespace object. It is used to update the class parameters with the values provided in the command-line arguments.

        Args:
            args (argparse.Namespace): Argument namespace containing the parameters.
        """
        self._params = vars(args)

    def count_alleles(self):
        """Count alleles for each SNP across all samples.

        Returns:
            numpy.ndarray: 2D array of allele counts with shape (n_loci, 2).
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
            transform_only (bool): Whether to transform, but not fit. Default is False.

        Returns:
            numpy.ndarray: Imputed data.
        """
        if transform_only:
            if X.size == 0:
                raise InvalidInputShapeError(
                    "One of the input datasets was empty. Did you remember to set some values as unknown in the 'sample_data' coordinates file?"
                )
            imputed = self.simputer.transform(X)
        else:
            self.simputer = clone(self.simputer)
            imputed = self.simputer.fit_transform(X)  # ensure it's not fit.
        return imputed

    def sort_samples(self, sample_data_filename):
        """Load sample_data and popmap and sort to match VCF file.

        Args:
            sample_data_filename (str): Filename of the sample data file.

        Raises:
            InvalidSampleDataError: If the sample data file format is incorrect.
        """
        self.sample_data = read_csv_with_dynamic_sep(sample_data_filename)

        if self.sample_data.shape[1] != 3:
            msg = f"'sample_data' must be a tab-delimited file with three columns: sampleID, x, and y. 'x' and 'y' should be longitude and latitude. However, we detected {self.sample_data.shape[1]} columns."
            self.logger.error(msg)
            raise InvalidSampleDataError(msg)

        self.sample_data.columns = ["sampleID", "x", "y"]
        self.sample_data["sampleID2"] = self.sample_data["sampleID"]
        self.sample_data.set_index("sampleID", inplace=True)
        self.samples = np.array(self.samples).astype(str)
        self.sample_data = self.sample_data.reindex(np.array(self.samples))

        # Sample ordering check
        self._check_sample_ordering()

        # Ensure correct CRS.
        gdf = self.plotting.processor.to_geopandas(self.sample_data[["x", "y"]])
        self.locs = self.plotting.processor.to_numpy(gdf)

        self.mask = self.mask[self.filter_mask]
        self.mask = self.mask[self.sort_indices]
        self.samples = self.samples[self.filter_mask]
        self.samples = self.samples[self.sort_indices]
        self.is_missing = self.is_missing[:, self.filter_mask]
        self.is_missing = self.is_missing[:, self.sort_indices]
        self.genotypes_iupac = self.genotypes_iupac[:, self.filter_mask]
        self.genotypes_iupac = self.genotypes_iupac[:, self.sort_indices]

    def normalize_target(self, y, transform_only=False):
        """Normalize locations, ignoring NaN.

        Args:
            y (numpy.ndarray): Array of locations to normalize.
            transform_only (bool): Whether to transform without fitting. Default is False.

        Returns:
            numpy.ndarray: Normalized locations.
        """
        if self.verbose >= 1:
            self.logger.info("Normalizing coordinates...")

        if transform_only:
            y = self.norm.transform(y)
        else:
            y = self.norm.fit_transform(y)

        if self.verbose >= 1:
            self.logger.info("Done normalizing coordinates.")

        return y

    def _check_sample_ordering(self):
        """Validate sample ordering between 'sample_data' and VCF files.

        Raises:
            SampleOrderingError: If the sample ordering is invalid after filtering and sorting.
        """
        # Create a set from sample_data for intersection check
        sample_data_set = set(self.sample_data["sampleID2"])

        # Create mask for filtering samples present in both self.samples and self.sample_data
        mask = [sample in sample_data_set for sample in self.samples]

        # Filter self.samples to those present in both lists and create a corresponding index list
        filtered_samples = []
        sort_indices = (
            []
        )  # This will store the indices that will be used to sort downstream objects
        for idx, sample in enumerate(self.samples):
            if sample in sample_data_set:
                filtered_samples.append(sample)
                sort_indices.append(idx)

        # Filter and sort self.sample_data to only include rows with sampleID2 present in filtered_samples
        self.sample_data = self.sample_data[
            self.sample_data["sampleID2"].isin(filtered_samples)
        ]
        self.sample_data.sort_values("sampleID2", inplace=True)
        self.sample_data.reset_index(drop=True, inplace=True)

        # Sort filtered_samples to match the order in sorted self.sample_data
        filtered_samples_sorted = sorted(
            filtered_samples, key=lambda x: list(self.sample_data["sampleID2"]).index(x)
        )

        # Ensure that filtered_samples_sorted and self.sample_data are in the same order now
        if not all(
            self.sample_data["sampleID2"].iloc[i] == x
            for i, x in enumerate(filtered_samples_sorted)
        ):
            msg = "Invalid sample ordering after filtering and sorting."
            self.logger.error(msg)
            raise SampleOrderingError(msg)

        # Compute sort_indices from self.samples based on
        # sorted filtered_samples
        self.sort_indices = [
            np.where(self.samples == x)[0][0] for x in filtered_samples_sorted
        ]

        # Store the mask and indices for later use in subsetting and
        # reordering other numpy arrays or objects
        self.filter_mask = mask

        self.logger.info(
            "Sample ordering, filtering mask, and sorting indices stored successfully."
        )

    def snps_to_012(self, min_mac=2, max_snps=None, return_values=True):
        """Convert IUPAC SNPs to 012 encodings.

        Args:
            min_mac (int): Minimum minor allele count. Default is 2.
            max_snps (int, optional): Maximum number of SNPs to retain.
            return_values (bool): Whether to return encoded values. Default is True.

        Returns:
            numpy.ndarray: Encoded genotypes if return_values is True, otherwise updates internal state.
        """
        if self.verbose >= 1:
            self.logger.info("Converting SNPs to 012-encodings.")

        self.genotypes = self.genotypes[:, self.filter_mask]
        self.genotypes = self.genotypes[:, self.sort_indices]

        self.genotypes_enc = np.sum(
            self.genotypes, axis=-1, where=~np.all(np.isnan(self.genotypes))
        )

        self.all_missing_mask = np.all(np.isnan(self.genotypes_enc), axis=0)
        self.genotypes_enc = self.genotypes_enc[:, ~self.all_missing_mask]
        self.locs = self.locs[~self.all_missing_mask]

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

        self.logger.debug(
            f"Encoded Genotypes: {self.genotypes_enc.T}, Shape: {self.genotypes_enc.T.shape}"
        )

        if return_values:
            return self.genotypes_enc.T
        else:
            self.genotypes_enc = self.genotypes_enc.T

    def filter_gt(self, gt, min_mac, max_snps, allele_counts):
        """Filter genotypes based on minor allele count and random subsets (max_snps).

        Args:
            gt (numpy.ndarray): Genotypes to filter.
            min_mac (int): Minimum minor allele count.
            max_snps (int, optional): Maximum number of SNPs to retain.
            allele_counts (numpy.ndarray): Allele counts.

        Returns:
            numpy.ndarray: Filtered genotypes.
        """
        if min_mac > 1:
            mac = 2 * allele_counts[:, 2] + allele_counts[:, 1]
            gt = gt[mac >= min_mac, :]

        if max_snps is not None:
            gt = gt[
                np.random.choice(range(gt.shape[0]), max_snps, replace=False),
                :,
            ]

        return gt

    def _find_optimal_clusters(self, features, args):
        """Find optimnal number of clusters from input features.

        Args:
            features (numpy.ndarray): Input features.
            args (argparse.Namespace): Argument namespace containing additional parameters.

        Returns:
            optimal_k (int): Optimal number of clusters.
        """
        max_k = args.maxk
        silhouette_scores = []

        for k in range(3, max_k + 1):
            kmeans = KMeans(n_clusters=k, n_init="auto", random_state=args.seed)
            cluster_labels = kmeans.fit_predict(features)
            silhouette_scores.append(silhouette_score(features, cluster_labels))

        optimal_k = np.argmax(silhouette_scores) + 2  # +2 to get the K value
        return optimal_k

    def _determine_bandwidth(self, data):
        """
        Automatically determine the bandwidth for KernelDensity using cross-validation.

        Args:
            data (np.ndarray | list): The data to estimate density.

        Returns:
            float: Optimal bandwidth for KDE.
        """
        bandwidths = np.logspace(-1, 1, 20)  # Test a range of bandwidths
        grid = GridSearchCV(
            KernelDensity(kernel="gaussian", metric="haversine"),
            param_grid={"bandwidth": bandwidths},
            cv=5,
        )
        grid.fit(data)
        optimal_bandwidth = grid.best_params_["bandwidth"]

        if self.verbose >= 1:
            self.logger.info(
                f"Optimal bandwidth for KernelDensity: {optimal_bandwidth}"
            )
        return optimal_bandwidth

    def _determine_eps(self, data, k=5):
        """Automatically determine the epsilon parameter for DBSCAN using k-distance.

        Args:
            data (np.ndarray | list): The data to cluster.
            k (int): Number of nearest neighbors.

        Returns:
            float: Optimal eps for DBSCAN.
        """
        # Compute pairwise distances
        distances = pairwise_distances(data)
        k_distances = np.sort(distances, axis=1)[
            :, k
        ]  # k-th nearest neighbor distances
        elbow_point = np.percentile(
            k_distances, 95
        )  # Use the 95th percentile as heuristic

        if self.verbose >= 1:
            self.logger.info(f"Estimated epsilon (eps) for DBSCAN: {elbow_point}")
        return elbow_point

    # Adjust the splits
    def _adjust_splits(self, train_split, val_split):
        """Adjust train, validation, and test splits based on user input.


        Args:
            train_split (float): Proportion of the data to use for training.
            val_split (float): Proportion of the data to use for validation and test combined.

        Returns:
            tuple: Adjusted train, validation, and test splits.

        """
        if train_split >= 1.0 or val_split >= 1.0:
            msg = "train_split and val_split must each be less than 1.0."
            self.logger.error(msg)
            raise ValueError(msg)

        if train_split + val_split != 1.0:
            msg = "The sum of train_split and val_split must equal 1.0."
            self.logger.error(msg)
            raise ValueError(msg)

        # Remaining proportion after training split
        remaining = 1.0 - train_split

        # Divide remaining proportion into validation and test splits
        validation_split = (remaining * val_split) / 2
        test_split = (remaining * val_split) / 2

        return train_split, validation_split, test_split

    def split_train_test(self, train_split, val_split, seed, args):
        """Splits the data into training, validation, and test datasets.

        Args:
            train_split (float): Proportion of the data to use for training.
            val_split (float): Proportion of the data to use for validation.
            seed (int): Random seed for reproducibility.
            args (argparse.Namespace): Argument namespace containing additional parameters.
        """
        if self.verbose >= 1:
            self.logger.info(
                "Splitting data into train, validation, and test datasets."
            )

        weighted_sampler = self.get_sample_weights(self.y, args)

        weights = (
            np.ones(self.y.shape[0])
            if weighted_sampler is None
            else weighted_sampler.weights
        )

        self.samples_weight = weights

        if self.verbose >= 1:
            self.logger.info(
                "Splitting data into train, validation, and test datasets."
            )

        val_split /= 2
        val_size = val_split / (train_split + val_split)

        if train_split + val_split >= 1:
            raise ValueError("The sum of train_split and val_split must be < 1.")

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
            np.arange(self.X.shape[0]),
            train_size=train_split + val_split,
            random_state=seed,
            shuffle=True,
        )

        # Ensure no overlap between train/val and test sets
        assert not set(train_val_indices).intersection(
            test_indices
        ), "Data leakage detected between train + val and test sets!"

        optimal_k = self._find_optimal_clusters(y_train_val, args)
        kmeans = KMeans(n_clusters=optimal_k, random_state=seed)
        cluster_labels = kmeans.fit_predict(y_train_val)

        X_train_val_list, X_test_list, X_train_list, X_val_list = [], [], [], []
        y_train_val_list, y_test_list, y_train_list, y_val_list = [], [], [], []
        train_val_indices_list, val_indices_list, test_indices_list = [], [], []
        train_val_samples_list, test_samples_list = [], []
        train_val_weights_list, test_weights_list = [], []

        for cluster_id in range(optimal_k):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            np.random.shuffle(cluster_indices)
            cluster_size = len(cluster_indices)

            split_index = int(cluster_size * (train_split + val_split))

            train_val_indices_cluster = cluster_indices[:split_index]
            test_indices_cluster = cluster_indices[split_index:]

            X_train_val_list.append(self.X[train_val_indices_cluster])
            y_train_val_list.append(self.y[train_val_indices_cluster])
            train_val_indices_list.append(train_val_indices_cluster)
            train_val_samples_list.append(self.samples[train_val_indices_cluster])
            train_val_weights_list.append(weights[train_val_indices_cluster])

            X_test_list.append(self.X[test_indices_cluster])
            y_test_list.append(self.y[test_indices_cluster])
            test_indices_list.append(test_indices_cluster)
            test_samples_list.append(self.samples[test_indices_cluster])
            test_weights_list.append(weights[test_indices_cluster])

        X_train_val = np.concatenate(X_train_val_list, axis=0)
        y_train_val = np.concatenate(y_train_val_list, axis=0)
        train_val_indices = np.concatenate(train_val_indices_list, axis=0)

        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        test_indices = np.concatenate(test_indices_list, axis=0)
        test_samples = np.concatenate(test_samples_list, axis=0)
        test_weights = np.concatenate(test_weights_list, axis=0)

        (
            X_train,
            X_val,
            y_train,
            y_val,
            train_indices,
            val_indices,
        ) = train_test_split(
            X_train_val,
            y_train_val,
            train_val_indices,
            test_size=val_size,
            random_state=seed,
            shuffle=True,
        )

        # Ensure no overlap between train and val sets
        assert not set(train_indices).intersection(
            val_indices
        ), "Data leakage detected between train and val sets!"

        if args.use_gradient_boosting:
            raise NotImplementedError(
                "Gradient Boosting is not yet supported with clustering-based sampling."
            )
            (
                X_val,
                X_train_val,
                y_val,
                y_train_val,
                train_indices,
                train_val_indices,
            ) = train_test_split(X_val, y_val, val_indices, test_size=0.5)

        optimal_k = self._find_optimal_clusters(y_train_val, args)
        kmeans = KMeans(n_clusters=optimal_k, random_state=seed)
        cluster_labels = kmeans.fit_predict(y_train_val)

        X_train_list, y_train_list, train_indices_list = [], [], []
        X_val_list, y_val_list, val_indices_list = [], [], []
        train_samples_list, val_samples_list = [], []
        train_weights_list, val_weights_list = [], []

        for cluster_id in range(optimal_k):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            np.random.shuffle(cluster_indices)
            split_index = int(
                len(cluster_indices) * (train_split / (train_split + val_split))
            )
            train_indices_cluster = cluster_indices[:split_index]
            val_indices_cluster = cluster_indices[split_index:]

            X_train_list.append(self.X[train_indices_cluster])
            y_train_list.append(self.y[train_indices_cluster])
            train_indices_list.append(train_indices_cluster)
            train_samples_list.append(self.samples[train_indices_cluster])
            train_weights_list.append(weights[train_indices_cluster])

            X_val_list.append(self.X[val_indices_cluster])
            y_val_list.append(self.y[val_indices_cluster])
            val_indices_list.append(val_indices_cluster)
            val_samples_list.append(self.samples[val_indices_cluster])
            val_weights_list.append(weights[val_indices_cluster])

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        train_indices = np.concatenate(train_indices_list, axis=0)
        X_val = np.concatenate(X_val_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)
        val_indices = np.concatenate(val_indices_list, axis=0)
        train_samples = np.concatenate(train_samples_list, axis=0)
        val_samples = np.concatenate(val_samples_list, axis=0)
        train_weights = np.concatenate(train_weights_list, axis=0)
        val_weights = np.concatenate(val_weights_list, axis=0)

        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.pred_samples = self.all_samples[self.pred_indices]

        self.known_indices = np.concatenate([train_indices, val_indices, test_indices])
        self.known_indices.sort()

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
            "train_samples": self.train_samples,
            "val_samples": self.val_samples,
            "test_samples": self.test_samples,
            "pred_samples": self.pred_samples,
            "known_samples": self.all_samples[self.known_indices],
            "train_weights": train_weights,
            "val_weights": val_weights,
            "test_weights": test_weights,
        }

        if args.use_gradient_boosting:
            data["X_train_val"] = X_train_val
            data["y_train_val"] = y_train_val

        self.data.update(data)

        self.indices = {
            "train_val_indices": train_val_indices,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": test_indices,
            "pred_indices": self.pred_indices,
            "true_indices": self.true_indices,
            "known_indices": self.known_indices,
        }

        if args.use_gradient_boosting:
            self.indices["train_val_indices"] = train_val_indices

        if self.verbose >= 1:
            self.logger.info("Created train, validation, and test datasets.")

        gdf_train = self.plotting.processor.to_geopandas(y_train)
        gdf_val = self.plotting.processor.to_geopandas(y_val)
        gdf_test = self.plotting.processor.to_geopandas(y_test)
        y_train = self.plotting.processor.to_numpy(gdf_train)
        y_val = self.plotting.processor.to_numpy(gdf_val)
        y_test = self.plotting.processor.to_numpy(gdf_test)

        self.logger.debug(
            f"Data Dictionary Object Shapes: {{str(k): v.shape for k, v in self.data.items()}}"
        )
        self.logger.debug(
            f"All indices object shapes: {{str(k): v.shape for k, v in self.indices.items()}}"
        )

        # Visualize results
        self.plotting.plot_scatter_samples_map(
            y_train, y_test, dataset="test", hue1=train_weights, hue2=test_weights
        )
        self.plotting.plot_scatter_samples_map(
            y_train, y_val, dataset="val", hue1=train_weights, hue2=val_weights
        )

        self.logger.info(f"Train Dataset Size: {X_train.shape[0]}")
        self.logger.info(f"Validation Dataset Size: {X_val.shape[0]}")
        self.logger.info(f"Test Dataset Size: {X_test.shape[0]}")
        self.logger.info(f"Train Weights Size: {train_weights.shape[0]}")
        self.logger.info(f"Validation Weights Size: {val_weights.shape[0]}")
        self.logger.info(f"Test Weights Size: {test_weights.shape[0]}")
        self.logger.info(f"Known Indices Size: {self.known_indices.shape[0]}")
        self.logger.info(f"Train Indices Size: {train_indices.shape[0]}")
        self.logger.info(f"Validation Indices Size: {val_indices.shape[0]}")
        self.logger.info(f"Test Indices Size: {test_indices.shape[0]}")
        self.logger.info(f"True Indices Size: {self.true_indices.shape[0]}")
        self.logger.info(f"Predicted Indices Size: {self.pred_indices.shape[0]}")
        self.logger.info(f"Train Samples Size: {self.train_samples.shape[0]}")
        self.logger.info(f"Validation Samples Size: {self.val_samples.shape[0]}")
        self.logger.info(f"Test Samples Size: {self.test_samples.shape[0]}")
        self.logger.info(f"Predicted Samples Size: {self.pred_samples.shape[0]}")

    def map_outliers_through_filters(self, original_indices, filter_stages, outliers):
        """Maps outlier indices through multiple filtering stages back to the original dataset.

        Args:
            original_indices (np.ndarray): Array of original indices before any filtering.
            filter_stages (List[np.ndarray]): List of arrays of indices after each filtering stage.
            outliers (np.ndarray): Outlier indices in the most filtered dataset.

        Returns:
            np.ndarray: Mapped outlier indices in the original dataset.
        """
        current_indices = outliers
        for stage_indices in reversed(filter_stages):
            current_indices = stage_indices[current_indices]
        return original_indices[current_indices]

    def load_and_preprocess_data(self, args):
        """Wrapper method to load and preprocess data.

        Code execution order is listed below.

        Sample Data Loading and Sorting - Calls `self.sort_samples` with `args.sample_data` to load and sort sample data. This step involves reading sample data, presumably including their geographical locations, and aligning them with the genomic data.

        SNP Encoding Transformation - Transforms Single Nucleotide Polymorphisms (SNPs) into a 0,1,2 encoding format using self.snps_to_012, considering parameters like min_mac (minimum minor allele count) and max_snps (maximum number of SNPs).

        Validation of Feature and Target Lengths - Ensures that the feature data (X) and target data (y) have the same number of rows using self.validate_feature_target_len.

        Missing Data Imputation on full dataset - Imputes missing data in `self.genotypes_enc` using `self.impute_missing`.

        Data Embedding and Transformation - Performs an embedding transformation (like PCA) on the imputed data (X) using self.embed, with full_dataset_only=True and transform_only=False.

        Index Mask Setup - Defines masks for prediction (self.pred_mask) to identify samples with known and unknown locations. Sets up index masks for data using self.setup_index_masks.

        Outlier Detection (Conditional) - If args.detect_outliers is True, performs outlier detection using self.run_outlier_detection.

        Extract datasets - self.X, self.y, self.X_pred, self.true_idx, self.all_samples, self.samples, self.pred_samples, self.outlier_samples, self.non_outlier_samples = self.extract_datasets(all_outliers, args)

        Plotting Outliers (Conditional) - If outliers are detected, plots the outliers using self.plotting.plot_outliers.

        Data Normalization (Placeholder) - Normalizes the target data (y) using self.normalize_target with placeholder=True. This does nothing, unless `placeholder=False`.

        Splitting into Train, Test, and Validation Sets - Splits the dataset into training, validation, and testing sets using self.split_train_test.

        Feature Embedding for Train, Validation, and Test Sets (Conditional) - `args.embedding_type` is not `none`, embeds the features of training, validation, and test sets using `self.embed`.

        Logging Data Split and DataLoader Creation - Logs the completion of data splitting and the start of DataLoader creation, if verbosity is enabled.

        DataLoader Creation - Creates DataLoaders for training, validation, and test datasets using `self.call_create_dataloaders`. Additional DataLoader is created for gradient boosting if args.use_gradient_boosting is True. Logging Completion of Preprocessing Logs the successful completion of data loading and preprocessing, if verbosity is enabled.
        """

        self.plotting = PlotGenIE(
            "cpu",
            args.output_dir,
            args.prefix,
            args.basemap_fips,
            args.highlight_basemap_counties,
            args.shapefile,
            show_plots=args.show_plots,
            fontsize=args.fontsize,
            filetype=args.filetype,
            dpi=args.plot_dpi,
            remove_splines=args.remove_splines,
        )

        if self.verbose >= 1:
            self.logger.info("Loading and preprocessing input data...")

        # Load and sort sample data
        self.sort_samples(args.sample_data)

        self.snps_to_012(
            min_mac=args.min_mac, max_snps=args.max_SNPs, return_values=False
        )

        if self.debug:
            dfX = pd.DataFrame(self.genotypes_enc)
            dfy = pd.DataFrame(self.locs, columns=["x", "y"])
            df = pd.concat([dfX, dfy], axis=1)
            df.to_csv("geogenie/test/X_mod.csv", header=True, index=False)

        # Make sure features and target have same number of rows.
        self.validate_feature_target_len()

        # Impute missing data and embed.
        X = self.impute_missing(self.genotypes_enc)

        # Define true_indices and pred_indices
        # If users did not supply any unknowns, randomly choose here.
        if np.any(np.isnan(self.locs)):  # User supplied unknowns.
            self.pred_mask = ~np.isnan(self.locs).any(axis=1)
        else:  # User did not supply unknown values. Randomly Choose.
            self.generate_unknowns(
                p=args.prop_unknowns, seed=args.seed, verbose=args.verbose >= 1
            )

        self.logger.debug(
            f"Pred Mask: {self.pred_mask}, Pred Mask Shape: {self.pred_mask.shape}"
        )

        self.mask = self.mask[~self.all_missing_mask]

        self.logger.debug(f"Mask: {self.mask}, Mask Shape: {self.mask.shape}")

        X, indices, y, index = self.setup_index_masks(X)
        self.all_indices = indices.copy()
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

        all_outliers = None
        if args.detect_outliers:
            all_outliers = self.run_outlier_detection(
                args, X, indices, y, index, filter_stage_indices
            )

        # Here X has not been imputed.
        (
            self.X,  # Known features only
            self.y,  # Knowns labels only
            self.X_pred,  # Unknown features only
            self.true_idx,  # Known indices only
            self.all_samples,  # All samples.
            self.samples,  # Non-outliers + knowns
            self.pred_samples,  # Unknowns only.
            self.outlier_samples,  # Only outliers.
            self.non_outlier_samples,  # Only non-outliers.
        ) = self.extract_datasets(all_outliers, args)

        if args.detect_outliers:
            # Write outlier samples to file.
            outlier_fn = str(args.prefix) + "_detected_outliers.txt"
            outlier_dir = Path(args.output_dir, "data")
            with open(outlier_dir / outlier_fn, "w") as fout:
                self.outlier_samples.sort()
                outdata = "\n".join(self.outlier_samples)
                fout.write(outdata + "\n")

            if self.verbose >= 1:
                self.logger.info(
                    f"{self.X.shape[0]} samples remaining after removing {len(all_outliers)} outliers and {len(self.X_pred)} samples with unknown localities."
                )

            self.plotting.plot_outliers(self.mask, self.locs)

        # placeholder=True makes it not do transform.
        self.split_train_test(args.train_split, args.val_split, args.seed, args)

        if args.verbose >= 1 and args.embedding_type != "none":
            self.logger.info("Embedding input features...")

        self.data["y_train"] = self.normalize_target(self.data["y_train"])

        for k, v in self.data.items():
            if k.startswith("X"):
                tonly = False if k == "X_train" else True

                # Impute missing values.
                imputed = self.impute_missing(v, transform_only=tonly)

                self.data[k] = self.embed(
                    args,
                    X=imputed,
                    alg=args.embedding_type,
                    transform_only=tonly,
                )
            elif k.startswith("y"):
                tonly = False if k == "y_train" else True

                if tonly:
                    self.data[k] = self.normalize_target(v, transform_only=tonly)

        # For bootstrapping with unknown predictions.
        self.genotypes_enc_imp = self.simputer.transform(self.genotypes_enc)
        self.genotypes_enc_imp = self.embed(
            args, X=self.genotypes_enc_imp, alg=args.embedding_type, transform_only=True
        )

        self.logger.debug(
            f"Encoded and Imputed Genotypes: {self.genotypes_enc_imp}, Shape: {self.genotypes_enc_imp.shape}"
        )

        if args.verbose >= 1 and args.embedding_type != "none":
            self.logger.info("Finished embedding features!")

        if args.verbose >= 1:
            self.logger.info("Data split into train, val, and test sets.")
            self.logger.info("Creating DataLoader objects...")

        self.logger.debug(
            f"Training Features: {self.data['X_train']}, Training Features Shape: {self.data['X_train'].shape}"
        )
        self.logger.debug(
            f"Validation Features: {self.data['X_val']}, Validation Featurs Shape: {self.data['X_val'].shape}"
        )
        self.logger.debug(
            f"Test Features: {self.data['X_test']}, Test Features Shape: {self.data['X_test'].shape}"
        )

        self.logger.debug(
            f"Training Targets: {self.data['y_train']}, Training Target Shape: {self.data['y_train'].shape}"
        )
        self.logger.debug(
            f"Validation Targets: {self.data['y_val']}, Validation Target Shape: {self.data['y_val'].shape}"
        )
        self.logger.debug(
            f"Test Targets: {self.data['y_test']}, Test Target Shape: {self.data['y_test'].shape}"
        )

        # Creating DataLoaders
        self.train_loader = self.call_create_dataloaders(
            self.data["X_train"],
            self.data["y_train"],
            args,
            False,
            self.data["train_samples"],
            dataset="train",
        )
        self.val_loader = self.call_create_dataloaders(
            self.data["X_val"],
            self.data["y_val"],
            args,
            True,
            self.data["val_samples"],
            dataset="val",
        )
        self.test_loader = self.call_create_dataloaders(
            self.data["X_test"],
            self.data["y_test"],
            args,
            True,
            self.data["test_samples"],
            dataset="test",
        )

        # Plot dataset distributions.
        self.plotting.plot_data_distributions(
            self.data["X_train"], self.data["X_val"], self.data["X_test"]
        )

        self.plotting.plot_data_distributions(
            self.data["y_train"],
            self.data["y_val"],
            self.data["y_test"],
            is_target=True,
        )

        if args.use_gradient_boosting:
            self.train_val_loader = self.call_create_dataloaders(
                self.data["X_train_val"],
                self.data["y_train_val"],
                args,
                True,
                self.data["train_samples"] + self.data["val_samples"],
                dataset="train_val",
            )

        if args.verbose >= 1:
            self.logger.info("DataLoaders created succesfully!")
            self.logger.info("Data loading and preprocessing completed!")

        # For setting new prefix.
        self.n_samples = self.genotypes_enc.shape[0]
        self.n_loci = self.data["X_train"].shape[1]

        self.logger.info(
            f"Number of samples in train dataset: {self.data['X_train'].shape[0]}"
        )
        self.logger.info(
            f"Number of samples in validation dataset: {self.data['X_val'].shape[0]}"
        )
        self.logger.info(
            f"Number of samples in test dataset: {self.data['X_test'].shape[0]}"
        )

        self.logger.info(
            f"Number of samples in prediction dataset: {self.data['X_pred'].shape[0]}"
        )

        self.logger.info(
            f"Number of samples in train weights: {self.data['train_weights'].shape[0]}"
        )

        self.logger.info(
            f"Number of samples in validation weights: {self.data['val_weights'].shape[0]}"
        )

        self.logger.info(
            f"Number of samples in test weights: {self.data['test_weights'].shape[0]}"
        )

        return f"{args.prefix}_N{self.n_samples}_L{self.n_loci}"

    def generate_unknowns(self, p=0.1, seed=None, verbose=False):
        """Randomly choose unknown samples for prediction.

        Only gets used if user does not supply and unkowns.

        Args:
            p (float): Proportion of samples to randomly select for the unknown prediction dataset. Defaults to 0.1.
            seed (int or None): Random seed to use for the random choice generator. Defaults to None (no random seed supplied).
            verbose (bool): Whether in verbose mode. Defaults to False.
        """
        if verbose:
            msg = f"Unknown 'nan' values were not provided in the '--sample_data' file. Randomly selecting {p * 100} percent of the samples (N={p * len(self.locs)} samples) for the unknown prediction dataset."
            self.logger.info(msg)

        N = len(self.locs)  # Number of rows in pred_mask.
        self.pred_mask = np.ones(N, dtype=bool)
        rng = np.random.default_rng(seed=seed)
        pred_idx = rng.choice(
            a=N, size=int(np.ceil(p * N)), replace=False, shuffle=False
        )
        self.pred_mask[pred_idx] = False

    def extract_datasets(self, outliers, args):
        """Extracts and separates datasets into known and predicted sets based on the presence of missing data.

        Args:
            outliers (numpy.ndarray): Array of outlier indices.
            args (argparse.Namespace): User-supplied arguments.

        Returns:
            tuple: Extracted datasets and sample indices.
        """

        self.mask[np.isin(self.all_indices, self.pred_indices)] = False
        pred_mask = np.zeros(len(self.all_indices), dtype=bool)
        pred_mask[np.isin(self.all_indices, self.pred_indices)] = True

        if args.detect_outliers:
            outlier_mask = np.zeros_like(self.mask)
            outlier_mask[np.isin(self.all_indices, outliers)] = True
            self.mask[outlier_mask] = False  # remove outliers.
        else:
            outlier_mask = self.mask.copy()

        return (
            self.genotypes_enc[self.mask, :],  # Knowns only.
            self.locs[self.mask, :],  # Knowns only.
            self.genotypes_enc[pred_mask, :],  # Unknowns only.
            self.all_indices[self.mask],  # Knowns only.
            self.samples,  # All samples.
            self.samples[self.mask],  # no outliers + knowns.
            self.samples[pred_mask],  # outliers + unknowns only.
            self.samples[outlier_mask],  # only outliers
            self.samples[~outlier_mask],  # only non-outliers
        )

    def validate_feature_target_len(self):
        """Validate that the feature and target datasets have the same length.

        Raises:
            InvalidInputShapeError: If the shapes of the feature and target datasets do not match.
        """
        if self.genotypes_enc.shape[0] != self.locs.shape[0]:
            msg = f"Invalid input shapes for genotypes and coorindates. The number of rows (samples) must be equal, but got: {self.genotypes_enc.shape}, {self.locs.shape}"
            self.logger.error(msg)
            raise InvalidInputShapeError(self.genotypes_enc.shape, self.locs.shape)

    def setup_index_masks(self, X):
        """Sets up index masks for filtering the data.

        Args:
            X (numpy.ndarray): Feature data.

        Returns:
            tuple: Filtered feature data, indices, target data, and sample indices.
        """
        indices = np.arange(self.locs.shape[0])
        X = X[self.pred_mask, :]
        y = self.locs[self.pred_mask, :]

        self.samples = self.samples[~self.all_missing_mask]
        index = self.samples[self.pred_mask]  # Non-unknowns.

        # Store indices after filtering for nan
        return X, indices, y, index

    def run_outlier_detection(self, args, X, indices, y, index, filter_stage_indices):
        """Performs outlier detection using geographic and genetic criteria.

        Args:
            args (argparse.Namespace): User-supplied arguments.
            X (numpy.ndarray): Feature matrix.
            indices (numpy.ndarray): Indices of samples.
            y (numpy.ndarray): Target variable (coordinates).
            index (numpy.ndarray): Index array for samples.
            filter_stage_indices (List[np.ndarray]): List of filter stage indices.

        Returns:
            numpy.ndarray: Array of outlier indices.

        Raises:
            OutlierDetectionError: If an error occurs during outlier detection.
        """
        try:
            outlier_detector = GeoGeneticOutlierDetector(
                args,
                pd.DataFrame(X, index=index),
                pd.DataFrame(y, index=index),
                output_dir=args.output_dir,
                prefix=args.prefix,
                n_jobs=args.n_jobs,
                url=args.shapefile,
                buffer=0.1,
                show_plots=args.show_plots,
                seed=args.seed,
                debug=args.debug,
                verbose=args.verbose,
            )

            outliers = outlier_detector.composite_outlier_detection(
                sig_level=args.significance_level,
                maxk=args.maxk,
                min_nn_dist=args.min_nn_dist,
            )

            all_outliers = np.concatenate((outliers["geographic"], outliers["genetic"]))

            # Returns outiler indices, remapped.
            mapped_all_outliers = self.map_outliers_through_filters(
                indices, filter_stage_indices, all_outliers
            )

            # Remove mapped outliers from data
            self.mask[np.isin(self.all_indices, mapped_all_outliers)] = False

            return all_outliers
        except Exception as e:
            raise OutlierDetectionError(f"Error occurred during outlier detection: {e}")

    def call_create_dataloaders(self, X, y, args, is_val, sample_ids, dataset=None):
        """Helper method to create DataLoader objects for different datasets.

        Args:
            X (numpy.ndarray | list): Feature data.
            y (numpy.ndarray | None): Target data.
            args (argparse.Namespace): User-supplied arguments.
            is_val (bool): Whether the dataset is validation and test data. Default is False.
            sample_ids (List[str]): List of sample IDs. Default is None.
            dataset (str): Dataset type. Required keyword argument. Default is None.

        Returns:
            torch.utils.data.DataLoader: DataLoader object.

        Raises:
            ValueError: If sampleIDs are not provided.
        """
        if sample_ids is None:
            msg = "Sample IDs must be provided to create DataLoader objects."
            self.logger.error(msg)
            raise ValueError(msg)

        if dataset not in {"train", "val", "test", "train_val"}:
            msg = "Invalid dataset type. Must be one of 'train', 'val', 'test', or 'train_val'."
            self.logger.error(msg)
            raise ValueError(msg)

        return self.create_dataloaders(
            X,
            y,
            args.batch_size,
            args,
            is_val=is_val,
            sample_ids=sample_ids,
            dataset=dataset,
        )

    @np.errstate(all="warn")
    def embed(
        self,
        args,
        X=None,
        alg="pca",
        full_dataset_only=False,
        transform_only=False,
    ):
        """Embed SNP data using one of several dimensionality reduction techniques.

        Args:
            args (argparse.Namespace): User-supplied arguments.
            X (numpy.ndarray): Data to embed. If None, uses self.genotypes_enc. Default is None.
            alg (str): Algorithm to use. Default is 'pca'.
            full_dataset_only (bool): If True, only embed and return full dataset. Default is False.
            transform_only (bool): If True, only transform without fitting. Default is False.

        Returns:
            numpy.ndarray: Embedded data.

        Raises:
            EmbeddingError: If the optimal number of components cannot be estimated.
        """
        if X is None:
            X = self.genotypes_enc.copy()

        do_embed = True
        if alg.lower() == "polynomial":
            if args.polynomial_degree > 2:
                self.logger.warn(
                    "Setting 'polynomial_degree' > 2 can lead to extremely large computational overhead. Do so at your own risk!!!"
                )
            emb = PolynomialFeatures(args.polynomial_degree)
        elif alg.lower() == "pca":
            n_components = args.n_components
            if n_components is None and not transform_only:
                n_components = self.get_num_pca_comp(X)
                emb = PCA(n_components=n_components, random_state=args.seed)

                if n_components is None:
                    msg = "n_componenets could not be estimated for PCA embedding."
                    self.logger.error(msg)
                    raise EmbeddingError(msg)

                if args.verbose >= 1:
                    self.logger.info(
                        f"Optimal number of pca components: {n_components}"
                    )
        elif alg.lower() == "kernelpca":
            n_components = self.gen_num_pca_comp(X)
            emb = KernelPCA(
                n_components=n_components,
                kernel="rbf",
                remove_zero_eig=True,
                random_state=args.seed,
            )

            if n_components is None:
                msg = "n_components could not be estimated for kernelpca embedding."
                self.logger.error(msg)
                raise EmbeddingError(msg)

            if args.verbose >= 1:
                self.logger.info(
                    f"Optimal number of kernelpca components: {n_components}"
                )

        elif alg.lower() == "nmf":
            n_components, recon_error = self.find_optimal_nmf_components(
                X, 2, min(X.shape[1] // 4, 50)
            )

            self.plotting.plot_nmf_error(recon_error, n_components)

            emb = NMF(n_components=n_components, random_state=args.seed)

            if n_components is None:
                msg = "n_components could not be estimated for NMF embedding."
                self.logger.error(msg)
                raise EmbeddingError(msg)

            if args.verbose >= 1:
                self.logger.info(f"Optimal number of NMF components: {n_components}")

        elif alg.lower() == "mca":
            n_components = self.perform_mca_and_select_components(
                X, range(2, min(100, X.shape[1])), args.embedding_sensitivity
            )

            if n_components is None:
                msg = "n_components could not be estimated for MCA embedding."
                self.logger.error(msg)
                raise EmbeddingError(msg)

            if args.verbose >= 1:
                self.logger.info(f"Optimal number of MCA components: {n_components}")

            emb = MCA(
                n_components=n_components,
                n_iter=25,
                check_input=True,
                random_state=args.seed,
                one_hot=True,
            )
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
            if alg.lower() == "lle":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    grid.fit(X)
                self.emb = grid.best_estimator_
                if args.verbose >= 1:
                    self.logger.info(
                        f"Best parameters for lle embedding: {grid.best_params_}"
                    )
                    self.logger.info(
                        f"Best score for lle embedding: {grid.best_score_}"
                    )
            else:
                self.emb = emb

        if do_embed:
            if full_dataset_only:
                if alg != "mca":
                    X = self.emb.fit_transform(X)
                else:
                    self.emb.fit(X)
                    X = self.emb.transform(X)

                if isinstance(X, pd.DataFrame):
                    X = X.to_numpy()
                return X

            elif not transform_only:
                if alg != "mca":
                    X = self.emb.fit_transform(X)
                else:
                    self.emb.fit(X)
                    X = self.emb.transform(X)
                if isinstance(X, pd.DataFrame):
                    X = X.to_numpy()
                return X

            elif transform_only:
                X = self.emb.transform(X)
                if isinstance(X, pd.DataFrame):
                    X = X.to_numpy()
                return X
        else:
            if isinstance(X, pd.DataFrame):
                X = X.to_numpy()
            return X.copy()

    def perform_mca_and_select_components(self, data, n_components_range, S):
        """Perform MCA on the provided data and select the optimal number of components.

        Args:
            data (pd.DataFrame): The categorical data.
            n_components_range (list): The range of components to explore.
            S (float): Sensitivity setting for selecting optimal number of components.

        Returns:
            int: The optimal number of components.
        """
        mca = MCA(n_components=max(n_components_range), n_iter=5, one_hot=True)
        mca = mca.fit(data)
        cumulative_inertia = mca.cumulative_inertia_
        optimal_n = self.select_optimal_components(cumulative_inertia, S)

        self.plotting.plot_mca_curve(cumulative_inertia, optimal_n)
        return optimal_n

    def select_optimal_components(self, cumulative_inertia, S):
        """Select the optimal number of components based on explained inertia.

        Args:
            cumulative_inertia (list): The cumulative inertia for each component.
            S (float): Sensitivity setting for selecting optimal number of components.

        Returns:
            int: The optimal number of components.
        """
        kneedle = KneeLocator(
            range(cumulative_inertia.shape[0]),
            cumulative_inertia,
            curve="concave",
            direction="increasing",
            S=S,
        )

        optimal_n = kneedle.knee
        return optimal_n

    def find_optimal_nmf_components(self, data, min_components, max_components):
        """Find the optimal number of components for NMF based on reconstruction error.

        Args:
            data (np.array): The data to fit the NMF model.
            min_components (int): The minimum number of components to try.
            max_components (int): The maximum number of components to try.

        Returns:
            tuple: The optimal number of components and the reconstruction errors.
        """
        errors = []
        components_range = range(min_components, max_components + 1)

        for n in components_range:
            model = NMF(n_components=n, init="random", random_state=0)
            model.fit(data)
            errors.append(model.reconstruction_err_)

        optimal_n = components_range[np.argmin(errors)]
        return optimal_n, errors

    def get_num_pca_comp(self, x):
        """Get optimal number of PCA components.

        Args:
            x (numpy.ndarray): Dataset to fit PCA to.

        Returns:
            int: Optimal number of principal components to use.
        """
        pca = PCA().fit(x)

        try:
            vr = np.cumsum(pca.explained_variance_ratio_)
        except AttributeError:
            vr = np.cumsum(pca.eigenvalues_) / sum(pca.eigenvalues_)

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
        sample_ids=None,
        dataset=None,
    ):
        """Create dataloaders for training, testing, and validation datasets.

        Args:
            X (numpy.ndarray | list): X dataset. Train, test, or validation.
            y (numpy.ndarray | None): Target data (train, test, or validation). None for GNN.
            batch_size (int): Batch size to use with model.
            args (argparse.Namespace): User-supplied arguments.
            is_val (bool): Whether using validation/ test dataset. Otherwise should be training dataset. Default is False.
            sample_ids (List[str]): List of sample IDs. Default is None.
            dataset (str): Dataset type. Required keyword argument. Default is None.

        Returns:
            torch.utils.data.DataLoader: DataLoader object suitable for the specified model type.
        """
        if sample_ids is None:
            msg = "Sample IDs must be provided to create DataLoader objects."
            self.logger.error(msg)
            raise ValueError(msg)

        if dataset not in {"train", "val", "test", "train_val"}:
            msg = "Invalid dataset type. Must be one of 'train', 'val', 'test', or 'train_val'."
            self.logger.error(msg)
            raise ValueError(msg)

        args = [X, y]

        kwargs = (
            {"sample_ids": sample_ids}
            if is_val
            else {"sample_weights": self.data["train_weights"]}
        )

        dataset = CustomDataset(*args, **kwargs)

        # Create DataLoader
        kwargs2 = {"batch_size": batch_size}
        kwargs2["shuffle"] = False if is_val else True
        return DataLoader(dataset, **kwargs2)

    def get_sample_weights(self, y, args):
        """Gets inverse sample_weights based on sampling density.

        Uses scikit-learn KernelDensity with a grid search to estimate the optimal bandwidth for GeographicDensitySampler.

        Args:
            y (numpy.ndarray): Target values.
            args (argparse.Namespace): User-supplied arguments.

        Returns:
            GeographicDensitySampler: The weighted sampler with estimated sample weights.
        """
        # Initialize weighted sampler if class weights are used
        self.weighted_sampler = None
        self.densities = None
        self.samples_weight = torch.ones((y.shape[0],), dtype=self.dtype)
        if args.use_weighted in {"sampler", "loss", "both"}:
            if args.verbose >= 1:
                self.logger.info("Estimating sampling density weights...")
                self.logger.info(
                    "Searching for optimal weighted sampler kde bandwidth..."
                )

            # Weight by sampling density.
            weighted_sampler = GeographicDensitySampler(
                pd.DataFrame(y, columns=["x", "y"]),
                focus_regions=args.focus_regions,
                use_kmeans=args.use_kmeans,
                use_kde=args.use_kde,
                use_dbscan=args.use_dbscan,
                w_power=args.w_power,
                max_clusters=args.max_clusters,
                max_neighbors=args.max_neighbors,
                normalize=args.normalize_sample_weights,
                verbose=args.verbose,
                logger=self.logger,
                dtype=self.dtype,
            )

            if args.verbose >= 1:
                self.logger.info("Done estimating sample weights.")

            self.weighted_sampler = weighted_sampler
            self.samples_weight = weighted_sampler.weights
            self.samples_weight_indices = weighted_sampler.indices
            self.densities = self.weighted_sampler.density
        return self.weighted_sampler

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
