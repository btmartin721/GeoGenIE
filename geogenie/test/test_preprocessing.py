import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

from geogenie.utils.data_structure import DataStructure
from geogenie.utils.argument_parser import setup_parser

# Import the necessary modules or functions from your scripts
# from your_script import YourPreprocessingFunction
# from locator import LocatorPreprocessingFunction


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        # Initialize any data or parameters needed for the tests
        # This can include loading a dataset, setting parameters, etc.
        self.args = setup_parser(test_mode=True)
        self.sample_data = "~/Documents/wtd/GeoGenIE/data/wtd_coords_N1426.txt"
        self.vcffile = "data/phase6_gtseq_subset.vcf.gz"
        self.args.sample_data = self.sample_data
        self.args.embedding_type = "none"
        self.args.output_dir = "~/Documents/wtd/GeoGenIE/geogenie/test/output"
        self.args.train_split = 0.7

        Path(self.args.output_dir + "/plots").mkdir(exist_ok=True, parents=True)

        self.ds = DataStructure(self.vcffile, debug=True)
        self.ds.load_and_preprocess_data(self.args)

        self.X_train_orig, self.y_train_orig = self._read_data_files(
            "~/Documents/wtd/GeoGenIE/geogenie/test/train_data_orig.csv"
        )
        self.X_test_orig, self.y_test_orig = self._read_data_files(
            "~/Documents/wtd/GeoGenIE/geogenie/test/test_data_orig.csv"
        )
        self.X_val_orig, self.y_val_orig = self._read_data_files(
            "~/Documents/wtd/GeoGenIE/geogenie/test/val_data_orig.csv"
        )

        X_mod, self.y_mod = self._read_data_files(
            "~/Documents/wtd/GeoGenIE/geogenie/test/X_mod.csv"
        )

        self.X_mod = X_mod

        self.af = self._calculate_allele_frequencies(X_mod)

        self.X_train = X_mod[self.ds.indices["train_indices"]]
        self.y_train = self.ds.data["y_train"]
        self.X_test = X_mod[self.ds.indices["test_indices"]]
        self.y_test = self.ds.data["y_test"]
        self.X_val = X_mod[self.ds.indices["val_indices"]]
        self.y_val = self.ds.data["y_val"]

        self.X_train, train_mask = self._remove_all_missing_samples(self.X_train)
        self.X_test, test_mask = self._remove_all_missing_samples(self.X_test)
        self.X_val, val_mask = self._remove_all_missing_samples(self.X_val)
        self.X_train_orig, train_orig_mask = self._remove_all_missing_samples(
            self.X_train_orig
        )
        self.X_test_orig, test_orig_mask = self._remove_all_missing_samples(
            self.X_test_orig
        )
        self.X_val_orig, val_orig_mask = self._remove_all_missing_samples(
            self.X_val_orig
        )
        self.X_mod, X_mod_mask = self._remove_all_missing_samples(self.X_mod)

        self.X_train_imp = self._impute_missing_alleles(self.X_train, self.af)
        self.X_test_imp = self._impute_missing_alleles(self.X_test, self.af)
        self.X_val_imp = self._impute_missing_alleles(self.X_val, self.af)

        self.train_indices = self.ds.indices["train_indices"][~train_mask]
        self.test_indices = self.ds.indices["test_indices"][~test_mask]
        self.val_indices = self.ds.indices["test_indices"][~val_mask]
        self.train_indices_orig = self.ds.indices["train_indices"][~train_orig_mask]
        self.test_indices_orig = self.ds.indices["test_indices"][~test_orig_mask]
        self.val_indices_orig = self.ds.indices["test_indices"][~val_orig_mask]

        self.train_orig_mask = train_orig_mask
        self.test_orig_mask = test_orig_mask
        self.val_orig_mask = val_orig_mask

    def _remove_all_missing_samples(self, a):
        """Remove samples with only missing values."""
        mask = np.all(np.isnan(a))
        return np.squeeze(a[~mask]), mask

    def _read_data_files(self, fn):
        df = pd.read_csv(fn)
        dfy = df[["x", "y"]]
        df.drop(["x", "y"], axis=1, inplace=True)
        X, y = df.to_numpy(), dfy.to_numpy()
        return np.squeeze(X), np.squeeze(y)

    # replace missing sites with binomial(2,mean_allele_frequency)
    def _impute_missing_alleles(ac, genotypes, af):
        """
        Impute missing alleles in allele count data.

        Args:
            ac (numpy.ndarray): 2D array of allele counts.
            missingness (numpy.ndarray): 2D boolean array where True indicates missing data.
            af (numpy.ndarray): 1D array of allele frequencies.

        Returns:
            numpy.ndarray: Updated allele counts with imputed values for missing data.
        """
        genotypes = genotypes.copy()
        mask = np.all(np.isnan(genotypes))
        genotypes = genotypes[~mask]

        missingness = np.isnan(genotypes)

        # Identify indices where imputation is needed
        impute_idx = np.where(missingness)

        # Generate random binomial values for each missing point
        random_values = np.random.binomial(2, af[impute_idx[0]])

        # Assign these random values to the corresponding positions in the ac
        # array
        genotypes[impute_idx] = random_values

        return genotypes

    def _calculate_allele_frequencies(self, genotypes):
        """
        Calculate allele frequencies from genotype data.

        Args:
            genotypes (numpy.ndarray): A 2D numpy array of genotypes.
                                    0 = reference, 1 = heterozygous, 2 = alternate, np.nan = missing.

        Returns:
            numpy.ndarray: Array of allele frequencies.
        """
        genotypes = genotypes.copy()
        gentoypes = genotypes.T

        # Handle missing data: Set np.nan to -1 for ease of calculation
        genotypes = np.nan_to_num(genotypes, nan=-1)

        # Count alleles: 0 (reference), 1 (heterozygous), and 2 (alternate)
        ref_count = np.sum(genotypes == 0, axis=1)
        het_count = np.sum(genotypes == 1, axis=1)
        alt_count = np.sum(genotypes == 2, axis=1)

        # Total number of alleles observed, excluding missing data (-1)
        total_alleles = 2 * (ref_count + het_count + alt_count)

        # Calculate allele frequencies: frequency of alternate allele
        allele_freq = (het_count + 2 * alt_count) / total_alleles

        return allele_freq

    def assertArrayAlmostEqual(self, arr1, arr2, tol=0.02):
        """Custom assert method to compare two numpy arrays within a tolerance."""
        arr1 = arr1.astype(int)
        arr2 = arr2.astype(int)

        arr1 = np.squeeze(arr1)
        arr2 = np.squeeze(arr2)

        print(arr1)
        print(arr2)

        self.assertEqual(arr1.shape, arr2.shape, "Array shapes are different.")

        mae = mean_absolute_error(arr1, arr2)
        rmse = np.sqrt(mean_squared_error(arr1, arr2))
        correlation = np.corrcoef(arr1.ravel(), arr2.ravel())[0, 1]
        self.assertLessEqual(mae, tol, f"MAE {mae} exceeds tolerance {tol}")
        self.assertLessEqual(rmse, tol, f"RMSE {rmse} exceeds tolerance {tol}")
        self.assertGreaterEqual(
            correlation, 1 - tol, f"Correlation {correlation} is less than expected"
        )

    def test_snps_encoding(self):
        # Test SNP encoding functionality
        # Compare the outputs
        self.assertArrayAlmostEqual(self.X_train_imp, self.X_train_orig, tol=0.02)

        self.assertArrayAlmostEqual(self.X_test_imp, self.X_test_orig, tol=0.02)
        self.assertArrayAlmostEqual(self.X_val_imp, self.X_val_orig, tol=0.02)

    def test_missing_data_imputation(self):
        # Test missing data imputation functionality
        X_mod_imp = self._impute_missing_alleles(self.X_mod, self.af)
        # X_test_imp = self._impute_missing_alleles(self.X_test, self.af)
        # X_val_imp = self._impute_missing_alleles(self.X_val, self.af)

        locs = np.vstack([self.y_train_orig, self.y_val_orig, self.y_test_orig])

        X_orig = np.vstack([self.X_train_orig, self.X_val_orig, self.X_test_orig])

        loc_mask = np.all(np.isnan(locs), axis=0)
        X_orig = X_orig[~loc_mask]
        X_mod_imp = X_mod_imp[~loc_mask]

        loc_mask2 = np.all(np.isnan(self.y_mod), axis=0)
        X_mod_imp = X_mod_imp[~loc_mask2]

        self.assertArrayAlmostEqual(X_mod_imp, X_orig, tol=0.02)
        # self.assertArrayAlmostEqual(X_test_imp, self.X_test_orig, tol=0.02)
        # self.assertArrayAlmostEqual(X_val_imp, self.X_val_orig, tol=0.02)

    # def test_target_normalization(self):
    #     pass
    # Test target normalization functionality
    # Similar structure to test_snps_encoding

    # Add more test cases for other preprocessing stages as necessary


if __name__ == "__main__":
    unittest.main()
