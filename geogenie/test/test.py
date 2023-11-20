import logging
import os
import random
import unittest
import sys

import pandas as pd
import numpy as np
import msprime

from geogenie import GeoGenIE
from geogenie.models.models import MLPRegressor
from geogenie.utils.argument_parser import setup_parser
from geogenie.utils.logger import setup_logger


class TestGeoGenIE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This method will run once before all tests
        cls.original_argv = sys.argv

    def setUp(self):
        self.original_argv = sys.argv
        # Modify sys.argv to include the path to your config file
        sys.argv = ["run_geogenie.py", "--config", "config_testing.yaml"]

        args = setup_parser()
        logfile = os.path.join(
            args.output_dir, "logfiles", f"{args.prefix}_logfile.txt"
        )
        setup_logger(logfile)
        self.geo_genie = GeoGenIE(args)

    @classmethod
    def tearDownClass(cls):
        # Restore the original sys.argv after all tests
        # This prevents potential issues with future tests.
        sys.argv = cls.original_argv

    def test_load_data(self):
        # Test the load_data method
        self.geo_genie.load_data()

        # Assertions to check if data is loaded correctly
        self.assertIsNotNone(self.geo_genie.genotypes)

    def test_optimize_parameters(self):
        # optimize_parameters returns a dictionary of best parameters
        best_params = self.geo_genie.optimize_parameters(
            criterion=GeoGenIE.euclidean_distance_loss, ModelClass=MLPRegressor
        )
        # Assertions to check if parameters are optimized
        self.assertIsInstance(best_params, dict)

    def test_with_config(self):
        self.test_argv = ["test.py", "--config", "config_testing.yaml"]
        self.setUp()  # Re-run setup with new argv

        self.tearDown()  # Clean up after the test

    def test_train_model(self):
        """Test the train_model method."""
        # This could involve creating a small test dataset and checking if the
        #  model trains
        result = self.geo_genie.train_model(...)
        # Assertions to check if training is successful
        self.assertIsNotNone(result)

    def test_optimize_parameters(self):
        # optimize_parameters returns a dictionary of best parameters
        best_params = self.geo_genie.optimize_parameters(
            criterion=GeoGenIE.euclidean_distance_loss, ModelClass=MLPRegressor
        )
        # Assertions to check if parameters are optimized
        self.assertIsInstance(best_params, dict)

    def simulate_vcf_file(self):
        # Constants
        num_loci = 50
        num_samples = 100
        missing_data_ratio = 0.2
        num_missing_values = int(num_samples * num_loci * missing_data_ratio)

        # Generate random positions for loci
        positions = sorted(random.sample(range(1, 100001), num_loci))

        # msprime simulation
        demography = msprime.Demography()
        demography.add_population(name="Pop1", initial_size=1000)
        demography.add_population(name="Pop2", initial_size=1000)
        demography.add_population_split(
            time=100, derived=["Pop1", "Pop2"], ancestral="AncestralPop"
        )

        samples = {f"Pop{k}": v for k, v in zip(range(1, 9), [num_samples // 8] * 8)}

        print(samples)
        sys.exit()

        # Simulate the tree sequence
        tree_sequence = msprime.sim_ancestry(
            samples={
                "Pop1": num_samples // 8,
                "Pop2": num_samples // 8,
                "Pop3": num_samples // 8,
                "Pop4": num_samples // 8,
            },
            demography=demography,
            sequence_length=max(positions),
            recombination_rate=1e-8,
            record_full_arg=True,
        )

        # Add mutations
        tree_sequence = msprime.sim_mutations(
            tree_sequence=tree_sequence, rate=1.5e-9, random_seed=42
        )

        # Create VCF dataframe
        vcf_df = pd.DataFrame(index=range(num_loci))
        vcf_df["#CHROM"] = [f"CM{str(i+1).zfill(4)}.1" for i in range(num_loci)]
        vcf_df["POS"] = positions
        vcf_df["ID"] = [f"rs{i+1}" for i in range(num_loci)]
        vcf_df["REF"] = [random.choice(["A", "T", "C", "G"]) for _ in range(num_loci)]
        vcf_df["ALT"] = [random.choice(["A", "T", "C", "G"]) for _ in range(num_loci)]
        vcf_df["QUAL"] = ["."] * num_loci
        vcf_df["FILTER"] = ["PASS"] * num_loci
        vcf_df["INFO"] = ["DP=100"] * num_loci
        vcf_df["FORMAT"] = ["GT:AD:DP:GQ"] * num_loci

        # Extract genotypes from tree sequence
        genotype_matrix = tree_sequence.genotype_matrix().T[:num_samples, :num_loci]

        # Introduce missing data
        for _ in range(num_missing_values):
            sample_idx = random.randint(0, num_samples - 1)
            locus_idx = random.randint(0, num_loci - 1)
            genotype_matrix[sample_idx, locus_idx] = -1  # -1 indicates missing data

        # Convert genotypes to VCF format
        for i in range(num_samples):
            sample_name = f"sample{i+1}"
            vcf_df[sample_name] = [
                "./." if g == -1 else f"{g}|{g}:1,1:2:99" for g in genotype_matrix[i, :]
            ]

        # Set minor allele counts for the first 10 loci
        for i in range(5):
            vcf_df.loc[i, "ALT"] = "1"
        for i in range(5, 10):
            vcf_df.loc[i, "ALT"] = "2"

        # Rest of the loci with varied MAC
        for i in range(10, num_loci):
            vcf_df.loc[i, "ALT"] = str(random.randint(3, 10))

        # Display the first few rows of the VCF DataFrame
        print(vcf_df.head())


if __name__ == "__main__":
    unittest.main()
