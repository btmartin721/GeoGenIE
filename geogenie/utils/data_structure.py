import logging
import os

import numpy as np
import pandas as pd
from pysam import VariantFile
from sklearn.model_selection import train_test_split
from torch import float as torchfloat
from torch import tensor as torchtensor
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from geogenie.samplers.samplers import Samplers
from geogenie.utils.gtseq2vcf import GTseqToVCF


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
        self.genotypes, self.is_missing = self._parse_genotypes()
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
        is_missing_list = []

        for record in self.vcf:
            if self.is_biallelic(record):
                genos = []
                missing = []
                for sample in record.samples.values():
                    genotype = sample.get("GT", (None, None))
                    genos.append(genotype)
                    missing.append(any(allele is None for allele in genotype))
                genotypes_list.append(genos)
                is_missing_list.append(missing)

        # Convert lists to NumPy arrays for efficient computation
        genotypes_array = np.array(genotypes_list, dtype="object")
        is_missing_array = np.array(is_missing_list, dtype=bool)
        return genotypes_array, is_missing_array

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

    def impute_missing(self, af_threshold=0.5):
        """Impute missing genotypes based on a given allele frequency threshold."""
        if self.verbose >= 1:
            self.logger.info("Imputing missing values.")
        # Example implementation (can be adapted)
        for i in range(self.genotypes.shape[0]):
            for j in range(self.genotypes.shape[1]):
                if self.is_missing[i, j]:
                    af = np.random.binomial(1, af_threshold, 2)
                    self.genotypes[i, j] = tuple(af)
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
        self.locs = np.array(
            [
                [
                    (x[0] - self.meanlong) / self.sdlong,
                    (x[1] - self.meanlat) / self.sdlat,
                ]
                for x in self.locs
            ]
        )

        if self.verbose >= 1:
            self.logger.info("Done normalizing.")

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

    def snps_to_012(self, min_mac=2, max_snps=None, return_values=True):
        if self.verbose >= 1:
            self.logger.info("Converting SNPs to 012-encodings.")

        self.genotypes_enc = np.sum(self.genotypes, axis=-1).astype("int8")

        allele_counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=3),
            axis=1,
            arr=self.genotypes_enc,
        )

        if min_mac > 1:
            mac = 2 * allele_counts[:, 2] + allele_counts[:, 1]
            self.genotypes_enc = self.genotypes_enc[mac >= min_mac, :]

        if max_snps is not None:
            self.genotypes_enc = self.genotypes_enc[
                np.random.choice(
                    range(self.genotypes_enc.shape[0]), max_snps, replace=False
                ),
                :,
            ]

        if self.verbose >= 1:
            self.logger.info("Input SNP data converted to 012-encodings.")

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

        # Prepare genotype and location data for training, validation, testing,
        # and prediction
        X_train = np.transpose(self.genotypes_enc[:, train_indices])
        X_val = np.transpose(self.genotypes_enc[:, val_indices])
        X_test = np.transpose(self.genotypes_enc[:, test_indices])
        X_pred = np.transpose(self.genotypes_enc[:, pred_indices])
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
            self.impute_missing()

        self.snps_to_012(
            min_mac=args.min_mac, max_snps=args.max_SNPs, return_values=False
        )

        self.normalize_locs()
        self.split_train_test(args.train_split, args.val_split, args.seed)

        if args.verbose >= 1:
            self.logger.info("Data split into train, val, and test sets.")
            self.logger.info("Creating DataLoader objects.")

        # Creating DataLoaders
        self.train_loader = self.create_dataloaders(
            self.data["X_train"],
            self.data["y_train"],
            args.class_weights,
            args.batch_size,
            y_strat=self.popmap_data,
            indices=self.indices["train_indices"],
        )

        # Creating DataLoaders
        self.val_loader = self.create_dataloaders(
            self.data["X_val"],
            self.data["y_val"],
            args.class_weights,
            args.batch_size,
        )

        # Creating DataLoaders
        self.test_loader = self.create_dataloaders(
            self.data["X_test"],
            self.data["y_test"],
            args.class_weights,
            args.batch_size,
        )

        if args.verbose >= 1:
            self.logger.info("DataLoaders created succesfully.")

        if args.verbose >= 1:
            self.logger.info("Data loading and preprocessing completed.")

    def create_dataloaders(
        self,
        X,
        y,
        class_weights,
        batch_size,
        y_strat=None,
        indices=None,
    ):
        """
        Create dataloaders for training, testing, and validation datasets.

        Args:
            X (numpy.ndarray): X dataset. Train, test, or validation.
            y (numpy.ndarray): Target data (train, test, or validation).
            class_weights (bool): If True, calculates class weights for weighted sampling.
            batch_size (int): Batch size to use with model.
            y_strat (dict): Mapping data used for class weights, if applicable.
            indices (numpy.ndarray): Indices to use if 'class_weights' is True.

        Returns:
            torch.utils.data.DataLoader: DataLoader object.
        """
        # Convert data to TensorDatasets
        dataset = TensorDataset(
            torchtensor(X, dtype=torchfloat),
            torchtensor(y, dtype=torchfloat),
        )

        weighted_sampler = None
        samples_weight = None
        # Initialize weighted sampler if class weights are used
        if class_weights:
            if y_strat is not None and indices is not None:
                samp = Samplers(indices)
                samples_weight = samp.get_class_weights_populations(y_strat)
                weighted_sampler = WeightedRandomSampler(
                    weights=samples_weight,
                    num_samples=len(samples_weight),
                    replacement=True,
                )
            elif y_strat is None and indices is not None:
                raise TypeError(
                    "y_strat and indices must both or neither be provided.",
                )
            elif y_strat is not None and indices is None:
                raise TypeError(
                    "y_strat and indices must both or neither be provided.",
                )

            self.samples_weight = samples_weight

        # Create DataLoaders
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=weighted_sampler,
        )

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
