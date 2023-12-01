import unittest
import numpy as np
from geogenie.outliers.detect_outliers import GeoGeneticOutlierDetector
from scipy.stats import gamma
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.decomposition import PCA


class TestGeoGeneticOutlierDetector(unittest.TestCase):
    def setUp(self):
        (
            self.genetic_data,
            self.geographic_data,
            self.cluster_centers,
        ) = self._create_realistic_data()

        (
            self.genetic_data,
            self.geographic_data,
        ) = self._insert_known_outliers(self.genetic_data, self.geographic_data, 3, 200)

        self.detector = GeoGeneticOutlierDetector(
            self.genetic_data, self.geographic_data
        )

    def _create_realistic_data(
        self, num_samples=50, num_loci=200, num_clusters=3, num_outliers=2
    ):
        np.random.seed(0)

        # Generate distinct clusters for genetic data
        cluster_centers = np.random.randint(0, 3, size=(num_clusters, num_loci))
        genetic_data = np.vstack(
            [
                center
                + np.random.randint(-1, 2, size=(num_samples // num_clusters, num_loci))
                for center in cluster_centers
            ]
        )

        # Generate corresponding geographic data for these clusters
        geographic_data = np.vstack(
            [
                np.random.multivariate_normal(
                    mean=[-115 + i * 5, 35 + i * 5],
                    cov=[[1, 0.5], [0.5, 1]],
                    size=(num_samples // num_clusters),
                )
                for i in range(num_clusters)
            ]
        )

        # Create genetic outliers: genetic profile from one cluster, placed
        # geographically in another
        genetic_outliers = np.vstack(
            [
                cluster_centers[(i + 1) % num_clusters]
                + np.random.randint(-1, 2, size=(1, num_loci))
                for i in range(num_outliers)
            ]
        )
        geographic_outliers = geographic_data[:num_outliers]

        # Create geographic outliers: geographic location from one cluster,
        # with genetic profile from another
        geographic_outliers_gen = geographic_data[-num_outliers:]
        genetic_outliers_gen = np.vstack(
            [
                cluster_centers[(i + 1) % num_clusters]
                + np.random.randint(-1, 2, size=(1, num_loci))
                for i in range(num_outliers)
            ]
        )

        # Combine normal samples and outliers
        genetic_data_outliers = np.vstack(
            [genetic_data, genetic_outliers, genetic_outliers_gen]
        )
        geographic_data_outliers = np.vstack(
            [geographic_data, geographic_outliers, geographic_outliers_gen]
        )

        return genetic_data_outliers, geographic_data_outliers, cluster_centers

    def _insert_known_outliers(
        self, genetic_data, geographic_data, num_clusters, num_outliers
    ):
        np.random.seed(0)  # Ensures reproducibility

        # Assuming the first num_clusters samples are the centroids of the clusters
        geographic_cluster_centers = geographic_data[:num_clusters]

        # Create outliers with genetic profiles from one cluster and geographic locations near another
        genetic_outliers = []
        geographic_outliers = []

        for i in range(num_outliers):
            # Select random genetic profile from a cluster
            genetic_cluster = np.random.choice(num_clusters)
            genetic_profile = genetic_data[
                np.random.choice(np.where(genetic_data[:, 0] == genetic_cluster)[0])
            ]

            # Select a different geographic cluster center for the outlier
            geographic_cluster = (genetic_cluster + 1) % num_clusters
            geographic_location = geographic_cluster_centers[
                geographic_cluster
            ] + np.random.normal(
                0, 0.01, size=(2,)
            )  # Small noise

            # Add the outliers to the lists
            genetic_outliers.append(genetic_profile)
            geographic_outliers.append(geographic_location)

        # Convert lists to numpy arrays and append to original data
        genetic_outliers = np.array(genetic_outliers)
        geographic_outliers = np.array(geographic_outliers)

        genetic_data_with_outliers = np.vstack((genetic_data, genetic_outliers))
        geographic_data_with_outliers = np.vstack(
            (geographic_data, geographic_outliers)
        )

        return genetic_data_with_outliers, geographic_data_with_outliers

    def _plot_coords(self, geographic_data_normal, geographic_data_outliers):
        # Convert geographic data to a GeoDataFrame

        num_outliers = 2

        gdf_normal = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(
                geographic_data_normal[:, 0], geographic_data_normal[:, 1]
            )
        )

        gdf_outliers = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(
                geographic_data_outliers[:, 0], geographic_data_outliers[:, 1]
            )
        )

        # Plotting the geographic data
        plt.figure(figsize=(10, 10))
        ax = gdf_normal.plot(marker="o", color="blue", markersize=5)
        ax = gdf_outliers.plot(ax=ax, marker="x", color="r", markersize=10)

        ax.set_title("Geographic Data Distribution", fontsize=18)
        ax.set_xlabel("Longitude", fontsize=18)
        ax.set_ylabel("Latitude", fontsize=18)
        plt.show()

    def _plot_pca(self, genetic_data_outliers, num_outliers=2):
        # Assuming the function _create_realistic_data is defined and used to
        # create the data

        # Perform PCA on the genetic data
        pca = PCA(n_components=2)
        genetic_data_pca = pca.fit_transform(genetic_data_outliers)

        # Visualizing the PCA
        plt.figure(figsize=(10, 6))
        plt.scatter(
            genetic_data_pca[:-num_outliers, 0],
            genetic_data_pca[:-num_outliers, 1],
            label="Normal Samples",
        )
        plt.scatter(
            genetic_data_pca[-num_outliers:, 0],
            genetic_data_pca[-num_outliers:, 1],
            label="Outliers",
            color="red",
        )
        plt.title("PCA of Genetic Data")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.show()

    def test_detect_outliers(self):
        # Assuming the GeoGeneticOutlierDetector class and other necessary functions are defined

        # Instantiate the detector with the dataset
        detector = GeoGeneticOutlierDetector(
            self.genetic_data,
            self.geographic_data,
            output_dir="geogenie_testoutliers8",
        )

        # Test the outlier detection
        outliers = detector.composite_outlier_detection(max_k=10)
        print(outliers)

        # self.assertIsInstance(genetic_outliers, np.ndarray)
        # self.assertIsInstance(geographic_outliers, np.ndarray)
        # self.assertListEqual(genetic_outliers.tolist(), [4, 7])
        # self.assertListEqual(geographic_outliers.tolist(), [48, 49])


if __name__ == "__main__":
    unittest.main()
