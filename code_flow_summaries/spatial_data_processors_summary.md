# Module: spatial_data_processors

## Imports

- warnings
- math
- pathlib
- geopandas
- numpy
- pandas
- requests
- geopy.distance
- scipy.spatial
- sklearn.cluster

## Functions

- __init__
- extract_basemap_path_url
- to_pandas
- to_numpy
- to_geopandas
- _ensure_is_pandas
- _ensure_is_numpy
- _ensure_is_gdf
- haversine_distance
- haversine_error
- calculate_statistics
- _validate_dists
- spherical_mean
- calculate_bounding_box
- nearest_neighbor
- calculate_convex_hull
- detect_clusters
- detect_outliers
- geodesic_distance

## Classes and Methods

- SpatialDataProcessor
  - __init__
  - extract_basemap_path_url
  - to_pandas
  - to_numpy
  - to_geopandas
  - _ensure_is_pandas
  - _ensure_is_numpy
  - _ensure_is_gdf
  - haversine_distance
  - haversine_error
  - calculate_statistics
  - _validate_dists
  - spherical_mean
  - calculate_bounding_box
  - nearest_neighbor
  - calculate_convex_hull
  - detect_clusters
  - detect_outliers
  - geodesic_distance