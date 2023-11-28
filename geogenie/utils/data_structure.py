import logging
import os

import numpy as np
import pandas as pd
from kneed import KneeLocator
from pysam import VariantFile
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, RobustScaler
from torch import float as torchfloat
from torch import tensor
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import DataLoader as GeoDataLoader

from geogenie.plotting.plotting import PlotGenIE
from geogenie.samplers.samplers import GeographicDensitySampler
from geogenie.utils.gtseq2vcf import GTseqToVCF
from geogenie.utils.utils import base_to_int, get_iupac_dict


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
                    genotype = sample.get("GT", (None, None))
                    genos.append(genotype)
                    missing.append(any(allele is None for allele in genotype))
                    alleles.append(sample.alleles)
                genotypes_list.append(genos)
                is_missing_list.append(missing)
                iupac_alleles.append(
                    self.map_alleles_to_iupac(alleles, get_iupac_dict())
                )

        # Convert lists to NumPy arrays for efficient computation
        genotypes_array = np.array(genotypes_list, dtype="object")
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

    def impute_missing(self, model_type, af_threshold=0.5):
        """Impute missing genotypes based on a given allele frequency threshold."""
        if self.verbose >= 1:
            self.logger.info("Imputing missing values.")
        # Example implementation (can be adapted)
        for i in range(self.genotypes.shape[0]):
            for j in range(self.genotypes.shape[1]):
                if self.is_missing[i, j]:
                    af = np.random.binomial(1, af_threshold, 2)
                    self.genotypes[i, j] = tuple(af)
        simputer = SimpleImputer(
            strategy="most_frequent",
            missing_values="N",
        )
        self.genotypes_iupac = simputer.fit_transform(
            self.genotypes_iupac.T,
        )
        self.genotypes_iupac = self.genotypes_iupac.T
        if self.verbose >= 1:
            self.logger.info("Imputations successful.")

    def sort_samples(self, sample_data_filename, class_weights, popmap=None):
        """Load sample_data and popmap and sort to match VCF file."""
        self.sample_data = pd.read_csv(sample_data_filename, sep="\t")
        self.sample_data["sampleID2"] = self.sample_data["sampleID"]
        self.sample_data.set_index("sampleID", inplace=True)
        self.samples = np.array(self.samples).astype(str)
        self.sample_data = self.sample_data.reindex(np.array(self.samples))

        if class_weights:
            if not popmap:
                raise FileNotFoundError(
                    "popmap file must be provided if class_weights argument is "
                    "provided."
                )
            else:
                self.popmap_data = pd.read_csv(
                    popmap, sep="\t", names=["sampleID", "populationID"]
                )
                self.popmap_data["sampleID2"] = self.popmap_data["sampleID"]
                self.popmap_data.set_index("sampleID", inplace=True)
                self.popmap_data = self.popmap_data.reindex(np.array(self.samples))
        else:
            self.popmap_data = None

        # Sample ordering check
        self._check_sample_ordering(class_weights)
        self.locs = np.array(self.sample_data[["x", "y"]])

    def normalize_locs(self):
        """Normalize locations, ignoring NaN."""
        if self.verbose >= 1:
            self.logger.info("Normalizing coordinates.")
        self.meanlong = np.nanmean(self.locs[:, 0])
        self.sdlong = np.nanstd(self.locs[:, 0])
        self.meanlat = np.nanmean(self.locs[:, 1])
        self.sdlat = np.nanstd(self.locs[:, 1])

        self.norm = MinMaxScaler()
        self.data["y_train"] = self.norm.fit_transform(self.data["y_train"])
        self.data["y_test"] = self.norm.transform(self.data["y_test"])
        self.data["y_val"] = self.norm.transform(self.data["y_val"])

        if self.verbose >= 1:
            self.logger.info("Done normalizing.")

    def _estimate_eps(self, y, n_neighbors=5):
        """
        Estimate an initial eps value for DBSCAN based on the average distance
        to the n-th nearest neighbor of each point.

        Args:
            y (numpy.ndarray): Array containing the 'x' and 'y' data (pre-normalized).
            n_neighbors (int): Number of neighbors to consider for each point.

        Returns:
            float: Estimated eps value.
        """
        coords = y.copy()
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(coords)
        distances, _ = nn.kneighbors(coords)

        # Use the average distance to the n-th nearest neighbor
        return np.mean(distances[:, n_neighbors - 1])

    def remove_outliers_dbscan(self, X, y, min_samples, args, dataset):
        """
        Remove outliers from a DataFrame using DBSCAN.

        Args:
            y (numpy.ndarray): Array containing the 'x' and 'y' data (pre-normalized).
            eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
            args (argparse.Namespace): Command-line argument namespace.
            dataset (str): One of "train", "test", "validation".

        Returns:
            numpy.ndarray: Mask with True values being non-outliers.
            numpy.ndarray: Array with outliers removed.
        """
        if args.verbose >= 1:
            self.logger.info(f"Removing potential outliers from {dataset} dataset...")

        genomic_reduced = pd.DataFrame(X)
        df_geo = pd.DataFrame(y, columns=["x", "y"])

        # Standardize both genomic and geographic data
        scaler = MinMaxScaler()
        geo_scaled = scaler.fit_transform(df_geo)
        genomic_scaled = scaler.fit_transform(genomic_reduced)

        # Combine the datasets
        combined_features = np.concatenate([geo_scaled, genomic_scaled], axis=1)

        combined_unscaled_y = np.concatenate(
            [df_geo.to_numpy(), genomic_scaled], axis=1
        )

        eps = self._estimate_eps(combined_features, n_neighbors=min_samples)
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(combined_features)

        # Labels of -1 indicate outliers
        mask = db.labels_ != -1

        plotting = PlotGenIE(
            "cpu",
            args.output_dir,
            args.prefix,
            show_plots=args.show_plots,
            fontsize=args.fontsize,
        )

        plotting.plot_dbscan_clusters(
            combined_unscaled_y,
            args.output_dir,
            args.prefix,
            dataset,
            db.labels_,
            show=args.show_plots,
        )

        if args.verbose >= 1:
            self.logger.info(
                f"Removed {np.count_nonzero(~mask)} outliers from {dataset} dataset."
            )

        return mask, df_geo[mask].to_numpy()

    def _check_sample_ordering(self, class_weights):
        """
        Check for sample and population ordering.
        """
        # Check for sample ordering
        for i, x in enumerate(self.samples):
            if self.sample_data["sampleID2"].iloc[i] != x:
                raise ValueError(
                    "Sample ordering failed! Check that sample IDs match the VCF."
                )

        # Check for population ordering
        if class_weights and self.popmap_data is not None:
            for i, x in enumerate(self.samples):
                if self.popmap_data["sampleID2"].iloc[i] != x:
                    raise ValueError(
                        "Population ordering failed! Check that the popmap "
                        "sample IDs match those in the VCF."
                    )

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
        if self.verbose >= 1:
            self.logger.info("Converting SNPs to 012-encodings.")

        self.genotypes_enc = np.sum(self.genotypes, axis=-1).astype("int8")

        allele_counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=3),
            axis=1,
            arr=self.genotypes_enc,
        )

        self.genotypes_enc = self.filter_gt(
            self.genotypes_enc, min_mac, max_snps, allele_counts
        )

        if self.verbose >= 1:
            self.logger.info("Input SNP data converted to 012-encodings.")

        if return_values:
            return self.genotypes_enc

    def filter_gt(self, gt, min_mac, max_snps, allele_counts):
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

    def snps_to_allele_counts(
        self,
        min_mac=1,
        max_snps=None,
        return_counts=True,
    ):
        """Filter and preprocess SNPs, then convert to allele counts."""
        if self.verbose >= 1:
            self.logger.info("Converting SNPs to allele counts.")

        # Apply Minor Allele Count (MAC) filter
        allele_counts = self.count_alleles()

        mac_filter = allele_counts[:, 1] >= min_mac
        filtered_genotypes = allele_counts[mac_filter]

        # Optionally subsample SNPs
        if max_snps is not None and max_snps < filtered_genotypes.shape[0]:
            selected_indices = np.random.choice(
                filtered_genotypes.shape[0], max_snps, replace=False
            )
            filtered_genotypes = filtered_genotypes[selected_indices]

        # Convert genotypes to allele counts
        allele_counts = filtered_genotypes

        self.genotypes_enc = allele_counts

        if self.verbose >= 1:
            self.logger.info("Successfully converted allele counts.")

        if return_counts:
            return allele_counts

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

        # Identify indices with non-NaN locations
        train_val_indices = np.argwhere(~np.isnan(self.locs[:, 0])).flatten()
        pred_indices = np.array(
            [x for x in range(len(self.locs)) if x not in train_val_indices]
        )

        # Split non-NaN samples into training + validation and test sets
        train_val_indices, test_indices = train_test_split(
            train_val_indices, train_size=train_split, random_state=seed
        )

        # Split training data into actual training and validation sets
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=val_split, random_state=seed
        )

        self.genotypes_enc = self.genotypes_enc.T

        # Prepare genotype and location data for training, validation, testing,
        # and prediction
        # X_train = np.transpose(self.genotypes_enc[:, train_indices])
        # X_val = np.transpose(self.genotypes_enc[:, val_indices])
        # X_test = np.transpose(self.genotypes_enc[:, test_indices])
        # X_pred = np.transpose(self.genotypes_enc[:, pred_indices])
        X_train = self.genotypes_enc[train_indices, :]
        X_val = self.genotypes_enc[val_indices, :]
        X_test = self.genotypes_enc[test_indices, :]
        X_pred = self.genotypes_enc[pred_indices, :]
        y_train = self.locs[train_indices]
        y_val = self.locs[val_indices]
        y_test = self.locs[test_indices]

        train_val_indices = train_val_indices
        train_indices = train_indices
        val_indices = val_indices
        test_indices = test_indices
        pred_indices = pred_indices

        self.data = {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "X_pred": X_pred,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        }

        self.indices = {
            "train_val_indices": train_val_indices,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": test_indices,
            "pred_indices": pred_indices,
        }

        if self.verbose >= 1:
            self.logger.info("Created train, validation, and test datasets.")

    def fetch(
        self,
        return_all=False,
        key=None,
        data_only=False,
        indices_only=False,
    ):
        if data_only and indices_only:
            raise ValueError(
                "data_only and indices_only cannot both be defined.",
            )

        if return_all and data_only:
            raise ValueError(
                "get_all and data_only cannot both be defined.",
            )

        if return_all and indices_only:
            raise ValueError(
                "get_all and indices_only cannot both be defined.",
            )

        if return_all:
            return self.data | self.indices  # merges the dicts.

        if data_only and key is None:
            return self.data
        elif data_only and key is not None:
            try:
                return self.data[key]
            except KeyError:
                self.logger.error(f"Invalid key when fetching data: {key}")
                raise KeyError(f"Invalid key when fetching data: {key}")
        elif indices_only and key is None:
            return self.indices
        elif indices_only and key is not None:
            try:
                return self.indices[key]
            except KeyError:
                self.logger.error(f"Invalid key when fetching indices: {key}")
                raise KeyError(f"Invalid key when fetching indices: {key}")
        elif key is not None:
            if key in self.data:
                return self.data[key]
            if key in self.indices:
                return self.indices[key]
        else:
            return self.data | self.indices  # Merge the two dicts.

    def load_and_preprocess_data(self, args):
        """Wrapper method to load and preprocess data."""

        if self.verbose >= 1:
            self.logger.info("Loading and preprocessing input data...")

        # Load and sort sample data
        self.sort_samples(args.sample_data, args.class_weights, args.popmap)

        # Replace missing data
        if args.impute_missing:
            self.impute_missing(args.model_type)

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

        self.split_train_test(args.train_split, args.val_split, args.seed)
        # X_train, X_test, X_val, _ = self.embed(
        #     args, n_components=6, alg="pca", return_values=True
        # )

        # mask_train, self.data["y_train"] = self.remove_outliers_dbscan(
        #     X_train,
        #     self.data["y_train"],
        #     int(X_train.shape[0] * args.outlier_detection_scaler),
        #     args,
        #     "train",
        # )
        # mask_test, self.data["y_test"] = self.remove_outliers_dbscan(
        #     X_test,
        #     self.data["y_test"],
        #     int(X_test.shape[0] * args.outlier_detection_scaler),
        #     args,
        #     "test",
        # )
        # mask_val, self.data["y_val"] = self.remove_outliers_dbscan(
        #     X_val,
        #     self.data["y_val"],
        #     int(X_val.shape[0] * args.outlier_detection_scaler),
        #     args,
        #     "validation",
        # )

        # self.data["X_train"] = self.data["X_train"][mask_train]
        # self.data["X_test"] = self.data["X_test"][mask_test]
        # self.data["X_val"] = self.data["X_val"][mask_val]
        # self.indices["train_indices"] = self.indices["train_indices"][mask_train]
        # self.indices["test_indices"] = self.indices["test_indices"][mask_test]
        # self.indices["val_indices"] = self.indices["val_indices"][mask_val]

        self.normalize_locs()
        self.embed(args, n_components=None, alg="pca", return_values=False)

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

    def embed(
        self, args, n_components=None, alg="polynomial", degree=2, return_values=False
    ):
        if args.model_type != "transformer":
            do_embed = True
        else:
            do_embed = False
        if alg.lower() == "polynomial":
            emb = PolynomialFeatures(degree)
        elif alg.lower() == "pca":
            if n_components is None:
                self.plotting = PlotGenIE(
                    "cpu",
                    args.output_dir,
                    args.prefix,
                    show_plots=args.show_plots,
                    fontsize=args.fontsize,
                )
                n_components = self.get_num_pca_comp(self.data["X_train"], args)
            emb = PCA(n_components=n_components)
        elif alg.lower() == "tsne":
            emb = TSNE(n_components=n_components)
        elif alg.lower() == "none":
            do_embed = False
        else:
            raise ValueError(f"Invalid 'alg' value pasesed to 'embed()': {alg}")

        if do_embed:
            if return_values:
                X_train = emb.fit_transform(self.data["X_train"])
                X_test = emb.transform(self.data["X_test"])
                X_val = emb.transform(self.data["X_val"])
                X_pred = emb.transform(self.data["X_pred"])
                return X_train, X_test, X_val, X_pred
            else:
                self.data["X_train"] = emb.fit_transform(self.data["X_train"])
                self.data["X_test"] = emb.transform(self.data["X_test"])
                self.data["X_val"] = emb.transform(self.data["X_val"])
                self.data["X_pred"] = emb.transform(self.data["X_pred"])

    def get_num_pca_comp(self, x, args):
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

        self.plotting.plot_pca_curve(
            x, vr, knee, args.output_dir, args.prefix, show=args.show_plots
        )

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
            y_strat (dict): Mapping data used for class weights, if applicable.
            indices (numpy.ndarray): Indices to use if 'class_weights' is True.
            model_type (str): Type of the model ('gcn', 'transformer', 'mlp').

        Returns:
            DataLoader: DataLoader object suitable for the specified model type.
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
            # Convert data to TensorDatasets
            dataset = TensorDataset(
                tensor(X, dtype=torchfloat),
                tensor(y, dtype=torchfloat),
            )

            # Custom sampler - density-based.
            # weighted_sampler = self.get_sample_weights(y, class_weights, args)
            weighted_sampler = self.get_sample_weights(
                self.norm.inverse_transform(y), class_weights, args
            )

            # Create DataLoaders
            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=weighted_sampler,
            )

    def get_sample_weights(self, y, class_weights, args):
        # Initialize weighted sampler if class weights are used
        weighted_sampler = None
        if class_weights:
            if args.verbose >= 1:
                self.logger.info("Estimating sampling density weights...")

            # Define the grid of bandwidths to search over
            bandwidths = np.linspace(0.01, 1.0, 30)

            if args.verbose == 2:
                self.logger.info("Searching for optimal kernel density bandwidth...")

            # Grid search with cross-validation
            grid = GridSearchCV(
                KernelDensity(kernel="gaussian"),
                {"bandwidth": bandwidths},
                cv=5,
                n_jobs=args.n_jobs,
                verbose=args.verbose,
            )
            grid.fit(y)

            if args.verbose == 2:
                self.logger.info("Done searching bandwidth.")

                # Optimal bandwidth
            best_bandwidth = grid.best_estimator_.bandwidth

            # Weight by sampling density.
            weighted_sampler = GeographicDensitySampler(
                y,
                bandwidth=best_bandwidth,
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

    @property
    def coord_scaler(self):
        """Get dictionary with coordinate summary stats for scaling.

        Returns:
            dict: Coordinate Mean and StdDev (longitude and latitude).
        """
        return {
            "meanlong": self.meanlong,
            "meanlat": self.meanlat,
            "sdlong": self.sdlong,
            "sdlat": self.sdlat,
        }
