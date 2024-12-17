# Module: detect_outliers

## Imports

- logging
- time
- os
- pathlib
- numpy
- pynndescent
- scipy.optimize
- scipy.spatial.distance
- scipy.stats
- geogenie.plotting.plotting
- geogenie.utils.scorers
- geogenie.utils.utils

## Functions

- __init__
- calculate_dgeo
- calculate_statistic
- rescale_statistic
- find_gen_knn
- find_geo_knn
- find_optimal_k
- predict_coords_knn
- fit_gamma_mle
- gamma_neg_log_likelihood
- multi_stage_outlier_knn
- analysis
- run_multistage
- filter_and_detect
- search_nn_optk
- plot_gamma_dist
- detect_outliers
- composite_outlier_detection

## Classes and Methods

- GeoGeneticOutlierDetector
  - __init__
  - calculate_dgeo
  - calculate_statistic
  - rescale_statistic
  - find_gen_knn
  - find_geo_knn
  - find_optimal_k
  - predict_coords_knn
  - fit_gamma_mle
  - gamma_neg_log_likelihood
  - multi_stage_outlier_knn
  - analysis
  - run_multistage
  - filter_and_detect
  - search_nn_optk
  - plot_gamma_dist
  - detect_outliers
  - composite_outlier_detection