import warnings
from math import atan2, cos, degrees, radians, sin, sqrt
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from geopy.distance import geodesic
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

warnings.simplefilter(action="ignore", category=FutureWarning)


class SpatialDataProcessor:
    """Spatial data processing class.

    This class provides a set of tools for processing spatial data, including calculating statistics, distances, and clustering.

    Attributes:
        tmpdir (Path): Temporary directory for shapefiles.
        output_dir (Path): Output directory for shapefiles.
        basemap_fips (str): FIPS code for the base map.
        crs (str): Coordinate reference system.
        logger (logging.Logger): Logger object.
    """

    def __init__(
        self, output_dir=None, basemap_fips=None, crs="EPSG:4326", logger=None
    ):
        """Instantiate the SpatialDataProcessor class.

        Args:
            output_dir (str, optional): Output directory for shapefiles. Defaults to None.
            basemap_fips (str, optional): FIPS code for the base map. Defaults to None.
            crs (str, optional): Coordinate reference system. Defaults to "EPSG:4326".
            logger (logging.Logger, optional): Logger object. Defaults to None.
        """
        self.tmpdir = None
        if output_dir is None:
            self.tmpdir = Path("./tmp_shapefiles")
            output_dir = self.tmpdir
        else:
            output_dir = Path(output_dir)

        self.output_dir = output_dir
        self.basemap_fips = basemap_fips
        self.crs = crs
        self.logger = logger

    def extract_basemap_path_url(self, url):
        """Extract base map from provided URL or file path.

        Args:
            url (str): URL or file path to extract base map from.

        Returns:
            str: Extracted base map path or URL.

        Raises:
            ValueError: If the base map FIPS code is not provided.
        """

        self.output_dir.mkdir(parents=True, exist_ok=True)
        fn = url.split("/")[-1]
        dest = self.output_dir / "shapefile" / fn

        try:
            # Log the download attempt
            if self.logger is not None:
                self.logger.info(f"Attempting to download file from {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            if self.logger is not None:
                self.logger.info(f"Successfully downloaded file to {dest}")
        except requests.RequestException as e:
            msg = f"Error downloading shapefile from {url}: {e}"
            if self.logger is not None:
                self.logger.error(msg)
            raise e

        if not Path(dest).is_file():
            msg = f"Could not find shapefile in provided path: {dest}"
            if self.logger is not None:
                self.logger.error(msg)
            raise FileNotFoundError(msg)

        if not Path(dest).exists():
            raise FileNotFoundError(f"Expected file does not exist: {dest}")

        try:
            if fn.endswith(".zip"):
                mapdata = gpd.read_file(f"zip://{dest}")
            else:
                mapdata = gpd.read_file(dest)
            return mapdata.to_crs(self.crs)
        except Exception as e:
            if self.logger is not None:
                self.logger.error(
                    f"Could not read map file {dest} from {url}. Error: {e}"
                )
            raise e

    def to_pandas(self, gdf):
        """Convert GeoPandas GeoDataFrame to Pandas DataFrame.

        Args:
            gdf (geopandas.GeoDataFrame): GeoDataFrame to convert to pandas.DataFrame.

        Returns:
            pandas.DataFrame: pandas DataFrame object.
        """
        gdf = self._ensure_is_gdf(gdf)
        df = pd.DataFrame(gdf.drop(columns="geometry"))
        df["x"] = gdf.geometry.x
        df["y"] = gdf.geometry.y
        return df

    def to_numpy(self, gdf):
        """Convert GeoPandas GeoDataFrame to NumPy array.

        Args:
            gdf (geopandas.GeoDataFrame): GeoDataFrame to convert to numpy.

        Returns:
            numpy.ndarray: Converted numpy array.
        """
        gdf = self._ensure_is_gdf(gdf)
        return np.vstack([gdf.geometry.x, gdf.geometry.y]).T

    def to_geopandas(self, df):
        """Convert DataFrame to GeoPandas GeoDataFrame with proper CRS.

        Args:
            df (pandas.DataFrame): DataFrame to convert to geopandas.GeoDataFrame.

        Returns:
            geopandas.GeoDataFrame: Converted GeoDataFrame object.
        """
        df = self._ensure_is_pandas(df)

        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"]))
        gdf = gdf.set_crs(self.crs)
        return gdf

    def _ensure_is_pandas(self, df):
        """Ensure the data is a Pandas DataFrame.

        Args:
            df (pandas.DataFrame): Data to validate.

        Returns:
            pandas.DataFrame: Validated DataFrame.

        Raises:
            TypeError: If the data is not a valid format.
        """
        if isinstance(df, pd.DataFrame):
            if "x" in df.columns and "y" in df.columns:
                return df.copy()
            else:
                msg = "geopandas.GeoDataFrame object missing 'x' and/ or 'y' column(s). Invalid format provided to SpatialDataProcessing."
                self.logger.error(msg)
                raise TypeError(msg)
        elif isinstance(df, np.ndarray) and df.shape[1] == 2:
            return pd.DataFrame(df, columns=["x", "y"])
        elif isinstance(df, gpd.GeoDataFrame):
            return self.to_pandas(df)
        else:
            msg = f"Invalid data structure provided to SpatialDataProcessor: {type(df)}"
            self.logger.error(msg)
            raise TypeError(msg)

    def _ensure_is_numpy(self, X):
        """Ensure the data is a NumPy array.

        Args:
            X (np.ndarray): Data to validate.

        Returns:
            np.ndarray: Validated NumPy array.

        Raises:
            TypeError: If the data is not a valid format.
        """
        if isinstance(X, gpd.GeoDataFrame):
            return self.to_numpy(X)
        elif isinstance(X, pd.DataFrame):
            return X.to_numpy()
        elif isinstance(X, np.ndarray) and X.shape[1] == 2:
            return X.copy()
        else:
            msg = f"Invalid data structure provided to SpatialDataProcessor: {type(X)}"
            self.logger.error(msg)
            raise TypeError(msg)

    def _ensure_is_gdf(self, gdf):
        """Ensure the data is a GeoPandas GeoDataFrame.

        Args:
            gdf (geopandas.GeoDataFrame): GeoDataFrame to validate.

        Returns:
            geopandas.GeoDataFrame: Validated GeoDataFrame.

        Raises:
            TypeError: If the data is not a valid format.
        """
        if not isinstance(gdf, gpd.GeoDataFrame):
            if isinstance(gdf, np.ndarray) and gdf.shape == 2:
                return self.to_geopandas(pd.DataFrame(gdf, columns=["x", "y"]))
            elif isinstance(gdf, pd.DataFrame):
                if "x" not in gdf.columns or "y" not in gdf.columns:
                    msg = "geopandas.GeoDataFrame object missing 'x' and/ or 'y' column(s). Invalid format provided to SpatialDataProcessing."
                    self.logger.error(msg)
                    raise TypeError(msg)
                return self.to_geopandas(gdf)
            else:
                msg = f"Invalid data structure provided to SpatialDataProcessor: {type(gdf)}"
                self.logger.error(msg)
                raise TypeError(msg)
        return gdf

    def haversine_distance(self, coords1, coords2):
        """Calculate haversine distance between two sets of points.

        Args:
            coords1 (np.ndarray): First set of coordinates.
            coords2 (np.ndarray): Second set of coordinates.

        Returns:
            np.ndarray: Haversine distance between coords1 and coords2.
        """

        c1 = self._ensure_is_numpy(coords1)
        c2 = self._ensure_is_numpy(coords2)

        R = 6371.0  # Earth's radius in kilometers
        lat1, lon1 = np.radians(c1[:, 0]), np.radians(c1[:, 1])
        lat2, lon2 = np.radians(c2[:, 0]), np.radians(c2[:, 1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    def haversine_error(self, point1, point2):
        """Calculate the haversine error between two geographic points.

        Args:
            point1 (float): First point to calculate error from.
            point2 (float): Second point to calculate error from.

        Returns:
            np.ndarray: Haversine error between points 1 and 2.
        """
        return self.haversine_distance(point1.y, point1.x, point2.y, point2.x)

    def calculate_statistics(self, gdf, max_boots=None, seed=None, known_coords=None):
        """Calculate statistics on properly projected data."""

        gdf = self._ensure_is_gdf(gdf)

        for sample_id, group in gdf.groupby("sampleID"):
            # Centroid calculations.
            mean_lon = group.dissolve().centroid.x
            mean_lat = group.dissolve().centroid.y

            median_lon = group.dissolve(aggfunc={"x": "median"}).centroid.x
            median_lat = group.dissolve(aggfunc={"y": "median"}).centroid.y

            if isinstance(mean_lat, pd.Series):
                mean_lat = mean_lat.iloc[0]
            if isinstance(mean_lon, pd.Series):
                mean_lon = mean_lon.iloc[0]
            if isinstance(median_lat, pd.Series):
                median_lat = median_lat.iloc[0]
            if isinstance(median_lon, pd.Series):
                median_lon = median_lon.iloc[0]

            std_dev_x, std_dev_y = group.geometry.x.std(), group.geometry.y.std()

            n = len(group)

            if max_boots is not None and n > max_boots:
                group = group.sample(n=max_boots, axis=0, random_state=seed)

            ci_95_x = 1.96 * std_dev_x / sqrt(n)
            ci_95_y = 1.96 * std_dev_y / sqrt(n)

            se_x = 1.96 * std_dev_x
            se_y = 1.96 * std_dev_y
            drms = sqrt(std_dev_x**2 + std_dev_y**2)

            resd = {
                "sampleID": sample_id,
                "x_mean": mean_lon,
                "y_mean": mean_lat,
                "x_median": median_lon,
                "y_median": median_lat,
                "std_dev_x": std_dev_x,
                "std_dev_y": std_dev_y,
                "ci_95_x": ci_95_x,
                "ci_95_y": ci_95_y,
                "se_x": se_x,
                "se_y": se_y,
                "drms": drms,
            }

            dfk = None
            if known_coords is not None:
                gdfk = self.to_geopandas(known_coords)
                gdfk = gdfk[gdfk["sampleID"] == sample_id]

                if gdfk.shape[0] > 1:
                    self.logger.warning(
                        f"Duplicate sampleIDs found in known coordinates file."
                    )
                    known_lat = gdfk.dissolve().centroid.y
                    known_lon = gdfk.dissolve().centroid.x
                elif not gdfk.empty:
                    known_lat = gdfk.geometry.y.iloc[0]
                    known_lon = gdfk.geometry.x.iloc[0]
                else:
                    self.logger.warning(
                        f"Known coordinates missing for sample {sample_id}, though they were expected."
                    )
                    gdfk = None

                if gdfk is not None:
                    dfk = self.to_pandas(gdfk)

                    haversine_err = self.haversine_distance(
                        np.array([[mean_lat, mean_lon]]),
                        np.array([[known_lat, known_lon]]),
                    )
                    geodesic_err = self.geodesic_distance(
                        np.array([[mean_lat, mean_lon]]),
                        np.array([[known_lat, known_lon]]),
                    )

                    haversine_err = self._validate_dists(haversine_err)
                    geodesic_err = self._validate_dists(geodesic_err)
                    resd["haversine_error"] = haversine_err
                    resd["geodesic_error"] = geodesic_err

            yield group, sample_id, dfk, resd

    def _validate_dists(self, err):
        if isinstance(err, (pd.Series, pd.DataFrame, np.ndarray)):
            if err.shape[0] == 1:
                if len(err.shape) == 1:
                    if isinstance(err, np.ndarray):
                        err = err[0]
                    else:
                        err = err.iloc[0]
                else:
                    raise ValueError(f"Distances are in invalid shape: {err.shape}")
        else:
            if not isinstance(err, float):
                if len(err) > 1:
                    err = np.mean(err)
                err = err[0]
        return err

    def spherical_mean(self, gdf):
        """Calculate spherical mean of geographic points."""

        gdf = self._ensure_is_gdf(gdf)

        x, y, z = 0, 0, 0
        for point in gdf.geometry:
            lat, lon = radians(point.y), radians(point.x)
            x += cos(lat) * cos(lon)
            y += cos(lat) * sin(lon)
            z += sin(lat)

        total = len(gdf)
        x /= total
        y /= total
        z /= total

        lon = atan2(y, x)
        hyp = sqrt(x * x + y * y)
        lat = atan2(z, hyp)

        return degrees(lat), degrees(lon)

    def calculate_bounding_box(self, gdf):
        """Calculate the bounding box for the dataset."""
        gdf = self._ensure_is_gdf(gdf)
        return gdf.total_bounds

    def nearest_neighbor(self, gdf):
        """Calculate the nearest neighbor for each point."""
        gdf = self._ensure_is_gdf(gdf)
        points = np.array([gdf.geometry.x, gdf.geometry.y]).T
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=2)
        nearest = distances[:, 1]
        return nearest

    def calculate_convex_hull(self, gdf):
        """Calculate the convex hull of all points."""
        gdf = self._ensure_is_gdf(gdf)
        return gdf.unary_union.convex_hull

    def detect_clusters(self, gdf, eps=0.5, min_samples=5):
        """Detect clusters using DBSCAN."""
        gdf = self._ensure_is_gdf(gdf)
        coords = np.array([gdf.geometry.x, gdf.geometry.y]).T
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        gdf["cluster"] = clustering.labels_
        return gdf

    def detect_outliers(self, gdf, threshold=2):
        """Identify outliers based on distance from the mean."""
        gdf = self._ensure_is_gdf(gdf)
        mean_point = gdf.geometry.unary_union.centroid
        distances = gdf.geometry.apply(lambda p: mean_point.distance(p))
        gdf["is_outlier"] = distances > distances.mean() + threshold * distances.std()
        return gdf

    def geodesic_distance(self, coords1, coords2):
        """Calculate geodesic distance between two sets of coordinates."""
        distances = [
            geodesic(tuple(coords1[i]), tuple(coords2[i])).km
            for i in range(len(coords1))
        ]
        return np.array(distances)
