# Master Code Flow Summary

## Table of Contents

- [Module: __init__](#__init__)
- [Module: module_summary_script](#module_summary_script)
- [Module: cli](#cli)
- [Module: geogenie](#geogenie)
- [Module: bootstrap](#bootstrap)
- [Module: optuna_opt](#optuna_opt)
- [Module: scorers](#scorers)
- [Module: transformers](#transformers)
- [Module: __init__](#__init__)
- [Module: logger](#logger)
- [Module: loss](#loss)
- [Module: utils](#utils)
- [Module: callbacks](#callbacks)
- [Module: exceptions](#exceptions)
- [Module: spatial_data_processors](#spatial_data_processors)
- [Module: data](#data)
- [Module: data_structure](#data_structure)
- [Module: argument_parser](#argument_parser)
- [Module: models](#models)
- [Module: __init__](#__init__)
- [Module: conf](#conf)
- [Module: detect_outliers](#detect_outliers)
- [Module: __init__](#__init__)
- [Module: plotting](#plotting)
- [Module: __init__](#__init__)
- [Module: interpolate](#interpolate)
- [Module: samplers](#samplers)

## Summary Table

| Module                  | Imports                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Functions                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Classes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|:------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| __init__                | geogenie.geogenie                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | None                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | None                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| module_summary_script   | pathlib, pandas, graphviz                                                                                                                                                                                                                                                                                                                                                                                                                                                              | parse_module_connections, extract_major_steps, extract_major_code_flow, save_major_steps_as_markdown, generate_major_code_flow_graph                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | None                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| cli                     | logging, pathlib, geogenie, geogenie.utils.argument_parser, geogenie.utils.logger                                                                                                                                                                                                                                                                                                                                                                                                      | main                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | None                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| geogenie                | csv, json, logging, os, time, traceback, functools, pathlib, numpy, optuna, pandas, torch, torch.nn, torch.optim, xgboost, scipy.stats, sklearn.metrics, geogenie.models.models, geogenie.optimize.bootstrap, geogenie.optimize.optuna_opt, geogenie.plotting.plotting, geogenie.samplers.interpolate, geogenie.samplers.samplers, geogenie.utils.callbacks, geogenie.utils.data, geogenie.utils.data_structure, geogenie.utils.loss, geogenie.utils.scorers, geogenie.utils.utils     | timer, save_execution_times, wrapper, __init__, total_execution_time_decorator, load_data, save_model, train_rf, visualize_oversampling, train_model, train_step, test_step, _batch_init, compute_rolling_statistics, predict_locations, calculate_prediction_metrics, _create_metrics_dictionary, get_all_stats, print_stats_to_logger, get_correlation_coef, plot_bootstrap_aggregates, perform_standard_training, extract_best_params, write_pred_locations, load_best_params, optimize_parameters, perform_bootstrap_training, evaluate_and_save_results, make_unseen_predictions, train_test_predict, wrapper, rescale_predictions, rescale_predictions, mad, coefficient_of_variation, within_threshold                                                                                              | GeoGenIE(__init__, total_execution_time_decorator, load_data, save_model, train_rf, visualize_oversampling, train_model, train_step, test_step, _batch_init, compute_rolling_statistics, predict_locations, calculate_prediction_metrics, _create_metrics_dictionary, get_all_stats, print_stats_to_logger, get_correlation_coef, plot_bootstrap_aggregates, perform_standard_training, extract_best_params, write_pred_locations, load_best_params, optimize_parameters, perform_bootstrap_training, evaluate_and_save_results, make_unseen_predictions, train_test_predict)                                                                                                                                                                                                                                             |
| bootstrap               | json, logging, os, threading, concurrent.futures, copy, pathlib, numpy, pandas, torch, torch, torch.utils.data, geogenie.plotting.plotting, geogenie.samplers.interpolate, geogenie.utils.callbacks, geogenie.utils.data, geogenie.utils.exceptions, geogenie.utils.utils                                                                                                                                                                                                              | __init__, _get_thread_local_rng, _resample_loaders, _resample_boot, reset_weights, reinitialize_model, train_one_bootstrap, bootstrap_training_generator, extract_best_params, save_bootstrap_results, perform_bootstrap_training, _process_boot_preds, _grouped_ci_boot, _bootrep_metrics_to_csv, _validate_sample_data                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Bootstrap(__init__, _get_thread_local_rng, _resample_loaders, _resample_boot, reset_weights, reinitialize_model, train_one_bootstrap, bootstrap_training_generator, extract_best_params, save_bootstrap_results, perform_bootstrap_training, _process_boot_preds, _grouped_ci_boot, _bootrep_metrics_to_csv, _validate_sample_data)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| optuna_opt              | json, logging, os, pickle, time, pathlib, numpy, optuna, pandas, torch, optuna, optuna.logging, torch, torch.utils.data, geogenie.plotting.plotting, geogenie.samplers.interpolate, geogenie.utils.callbacks, geogenie.utils.data, geogenie.utils.loss                                                                                                                                                                                                                                 | __init__, map_sampler_indices, objective_function, extract_features_labels, run_rf_training, run_training, set_gb_param_grid, set_param_grid, evaluate_model, perform_optuna_optimization, process_optuna_results, write_optuna_study_details, objective                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Optimize(__init__, map_sampler_indices, objective_function, extract_features_labels, run_rf_training, run_training, set_gb_param_grid, set_param_grid, evaluate_model, perform_optuna_optimization, process_optuna_results, write_optuna_study_details)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| scorers                 | logging, math, numba, numpy, scipy.stats, sklearn.manifold, sklearn.metrics, geogenie.utils.spatial_data_processors                                                                                                                                                                                                                                                                                                                                                                    | kstest, calculate_r2_knn, calculate_rmse, haversine_distance, predict, lle_reconstruction_scorer                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | LocallyLinearEmbeddingWrapper(predict, lle_reconstruction_scorer)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| transformers            | logging, numpy, scipy.sparse, sklearn.base, sklearn.preprocessing, sklearn.utils.extmath, sklearn.utils.validation                                                                                                                                                                                                                                                                                                                                                                     | __init__, fit, transform, _normalize_data, _compute_S_matrix, _store_results, __init__, fit, transform, inverse_transform                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | MCA(__init__, fit, transform, _normalize_data, _compute_S_matrix, _store_results), MinMaxScalerGeo(__init__, fit, transform, inverse_transform)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| __init__                | None                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | None                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | None                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| logger                  | logging, pathlib                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | setup_logger                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | None                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| loss                    | torch, torch.nn                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | weighted_rmse_loss, __init__, forward, __init__, forward                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | WeightedDRMSLoss(__init__, forward), WeightedHuberLoss(__init__, forward)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| utils                   | logging, signal, contextlib, io, numpy, pandas, torch, sklearn.cluster, geogenie.utils.exceptions                                                                                                                                                                                                                                                                                                                                                                                      | check_column_dtype, detect_separator, read_csv_with_dynamic_sep, time_limit, validate_is_numpy, assign_to_bins, geo_coords_is_valid, get_iupac_dict, signal_handler                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | None                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| callbacks               | logging, pathlib, numpy, torch                                                                                                                                                                                                                                                                                                                                                                                                                                                         | callback_init, __init__, __call__, save_checkpoint, load_best_model                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | EarlyStopping(__init__, __call__, save_checkpoint, load_best_model)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| exceptions              | None                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | __init__, __init__, __init__, __init__, __init__, __init__, __init__                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | GPUUnavailableError(__init__), ResourceAllocationError(__init__), TimeoutException(None), DataStructureError(None), InvalidSampleDataError(__init__), SampleOrderingError(__init__), InvalidInputShapeError(__init__), EmbeddingError(__init__), OutlierDetectionError(__init__)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| spatial_data_processors | warnings, math, pathlib, geopandas, numpy, pandas, requests, geopy.distance, scipy.spatial, sklearn.cluster                                                                                                                                                                                                                                                                                                                                                                            | __init__, extract_basemap_path_url, to_pandas, to_numpy, to_geopandas, _ensure_is_pandas, _ensure_is_numpy, _ensure_is_gdf, haversine_distance, haversine_error, calculate_statistics, _validate_dists, spherical_mean, calculate_bounding_box, nearest_neighbor, calculate_convex_hull, detect_clusters, detect_outliers, geodesic_distance                                                                                                                                                                                                                                                                                                                                                                                                                                                               | SpatialDataProcessor(__init__, extract_basemap_path_url, to_pandas, to_numpy, to_geopandas, _ensure_is_pandas, _ensure_is_numpy, _ensure_is_gdf, haversine_distance, haversine_error, calculate_statistics, _validate_dists, spherical_mean, calculate_bounding_box, nearest_neighbor, calculate_convex_hull, detect_clusters, detect_outliers, geodesic_distance)                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| data                    | torch, torch.utils.data                                                                                                                                                                                                                                                                                                                                                                                                                                                                | __init__, features, features, labels, labels, sample_weights, sample_weights, sample_ids, sample_ids, n_features, n_labels, __len__, __getitem__                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | CustomDataset(__init__, features, features, labels, labels, sample_weights, sample_weights, sample_ids, sample_ids, n_features, n_labels, __len__, __getitem__)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| data_structure          | logging, os, warnings, pathlib, numpy, pandas, pysam, torch, kneed, sklearn.base, sklearn.cluster, sklearn.decomposition, sklearn.impute, sklearn.manifold, sklearn.metrics, sklearn.model_selection, sklearn.neighbors, sklearn.preprocessing, torch.utils.data, geogenie.outliers.detect_outliers, geogenie.plotting.plotting, geogenie.samplers.samplers, geogenie.utils.data, geogenie.utils.exceptions, geogenie.utils.scorers, geogenie.utils.transformers, geogenie.utils.utils | __init__, _load_vcf_file, _decompress_recompress_index_and_load_vcf, _decompress_vcf, _recompress_vcf, _index_vcf, _cleanup_decompressed_file, _parse_genotypes, map_alleles_to_iupac, is_biallelic, define_params, count_alleles, impute_missing, sort_samples, normalize_target, _check_sample_ordering, snps_to_012, filter_gt, _find_optimal_clusters, _determine_bandwidth, _determine_eps, _adjust_splits, split_train_test, map_outliers_through_filters, load_and_preprocess_data, generate_unknowns, extract_datasets, validate_feature_target_len, setup_index_masks, run_outlier_detection, call_create_dataloaders, embed, perform_mca_and_select_components, select_optimal_components, find_optimal_nmf_components, get_num_pca_comp, create_dataloaders, get_sample_weights, params, params | DataStructure(__init__, _load_vcf_file, _decompress_recompress_index_and_load_vcf, _decompress_vcf, _recompress_vcf, _index_vcf, _cleanup_decompressed_file, _parse_genotypes, map_alleles_to_iupac, is_biallelic, define_params, count_alleles, impute_missing, sort_samples, normalize_target, _check_sample_ordering, snps_to_012, filter_gt, _find_optimal_clusters, _determine_bandwidth, _determine_eps, _adjust_splits, split_train_test, map_outliers_through_filters, load_and_preprocess_data, generate_unknowns, extract_datasets, validate_feature_target_len, setup_index_masks, run_outlier_detection, call_create_dataloaders, embed, perform_mca_and_select_components, select_optimal_components, find_optimal_nmf_components, get_num_pca_comp, create_dataloaders, get_sample_weights, params, params) |
| argument_parser         | argparse, ast, logging, os, warnings, yaml, torch.cuda, geogenie.utils.exceptions                                                                                                                                                                                                                                                                                                                                                                                                      | load_config, validate_positive_int, validate_positive_float, validate_gpu_number, validate_n_jobs, validate_split, validate_verbosity, validate_seed, validate_lower_str, setup_parser, validate_str2list, validate_dtype, validate_gb_params, validate_weighted_opts, validate_colorscale, validate_smote, validate_embeddings, validate_max_neighbors, validate_significance_levels, validate_inputs, __call__                                                                                                                                                                                                                                                                                                                                                                                           | EvaluateAction(__call__)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| models                  | logging, numpy, torch, torch.nn                                                                                                                                                                                                                                                                                                                                                                                                                                                        | __init__, _define_model, forward                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | MLPRegressor(__init__, _define_model, forward)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| __init__                | None                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | None                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | None                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| conf                    | os, sys                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | None                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | None                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| detect_outliers         | logging, time, os, pathlib, numpy, pynndescent, scipy.optimize, scipy.spatial.distance, scipy.stats, geogenie.plotting.plotting, geogenie.utils.scorers, geogenie.utils.utils                                                                                                                                                                                                                                                                                                          | __init__, calculate_dgeo, calculate_statistic, rescale_statistic, find_gen_knn, find_geo_knn, find_optimal_k, predict_coords_knn, fit_gamma_mle, gamma_neg_log_likelihood, multi_stage_outlier_knn, analysis, run_multistage, filter_and_detect, search_nn_optk, plot_gamma_dist, detect_outliers, composite_outlier_detection                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | GeoGeneticOutlierDetector(__init__, calculate_dgeo, calculate_statistic, rescale_statistic, find_gen_knn, find_geo_knn, find_optimal_k, predict_coords_knn, fit_gamma_mle, gamma_neg_log_likelihood, multi_stage_outlier_knn, analysis, run_multistage, filter_and_detect, search_nn_optk, plot_gamma_dist, detect_outliers, composite_outlier_detection)                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| __init__                | None                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | None                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | None                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| plotting                | logging, warnings, pathlib, geopandas, matplotlib, matplotlib.colors, matplotlib.lines, matplotlib.patches, matplotlib.pyplot, numpy, pandas, scipy.stats, seaborn, torch, kneed, optuna, optuna, pykrige.ok, scipy.stats, sklearn.exceptions, sklearn.linear_model, sklearn.pipeline, sklearn.preprocessing, geogenie.samplers.samplers, geogenie.utils.exceptions, geogenie.utils.spatial_data_processors, geogenie.utils.utils                                                      | __init__, plot_times, plot_smote_bins, _remove_spines, _plot_smote_scatter, plot_history, make_optuna_plots, plot_bootstrap_aggregates, update_metric_labels, update_config_labels, plot_scatter_samples_map, plot_geographic_error_distribution, _run_kriging, _set_cbar_fontsize, _plot_scatter_map, _make_colorbar, plot_cumulative_error_distribution, _fill_kde_with_gradient, plot_zscores, plot_error_distribution, polynomial_regression_plot, plot_mca_curve, plot_nmf_error, plot_pca_curve, plot_outliers, plot_gamma_distribution, plot_sample_with_density, _highlight_counties, visualize_oversample_clusters, plot_data_distributions, pfx, pfx, outdir, outdir, obp, obp, roundup, roundup, roundup, calculate_95_ci, roundup                                                              | PlotGenIE(__init__, plot_times, plot_smote_bins, _remove_spines, _plot_smote_scatter, plot_history, make_optuna_plots, plot_bootstrap_aggregates, update_metric_labels, update_config_labels, plot_scatter_samples_map, plot_geographic_error_distribution, _run_kriging, _set_cbar_fontsize, _plot_scatter_map, _make_colorbar, plot_cumulative_error_distribution, _fill_kde_with_gradient, plot_zscores, plot_error_distribution, polynomial_regression_plot, plot_mca_curve, plot_nmf_error, plot_pca_curve, plot_outliers, plot_gamma_distribution, plot_sample_with_density, _highlight_counties, visualize_oversample_clusters, plot_data_distributions, pfx, pfx, outdir, outdir, obp, obp)                                                                                                                       |
| __init__                | None                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | None                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | None                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| interpolate             | logging, copy, pathlib, matplotlib.pyplot, numpy, seaborn, torch, scipy.spatial.distance, sklearn.cluster, sklearn.metrics, sklearn.model_selection, sklearn.neighbors, torch.utils.data, geogenie.plotting.plotting, geogenie.utils.data                                                                                                                                                                                                                                              | run_genotype_interpolator, process_interp, resample_interp, reset_weighted_sampler, __init__, _determine_optimal_neighbors, _find_nearest_neighbors, interpolate_genotypes, _shuffle_over_sampled, _estimate_allele_frequencies, _vectorized_sample_hybrid, _assign_labels_to_synthetic_samples, _perform_kmeans_clustering, _calculate_centroids, _calculate_centroid_genotype, _calculate_optimal_bandwidth, _perform_density_estimation, _automated_parameter_tuning                                                                                                                                                                                                                                                                                                                                    | GenotypeInterpolator(__init__, _determine_optimal_neighbors, _find_nearest_neighbors, interpolate_genotypes, _shuffle_over_sampled, _estimate_allele_frequencies, _vectorized_sample_hybrid, _assign_labels_to_synthetic_samples, _perform_kmeans_clustering, _calculate_centroids, _calculate_centroid_genotype, _calculate_optimal_bandwidth, _perform_density_estimation, _automated_parameter_tuning)                                                                                                                                                                                                                                                                                                                                                                                                                 |
| samplers                | logging, os, jenkspy, matplotlib.pyplot, numpy, pandas, seaborn, torch, geopy.distance, imblearn.combine, imblearn.over_sampling, imblearn.under_sampling, scipy, scipy.spatial.distance, scipy.stats, sklearn.cluster, sklearn.metrics, sklearn.neighbors, sklearn.preprocessing, geogenie.utils.spatial_data_processors, geogenie.utils.utils, numpy, sklearn.cluster, sklearn.metrics, sklearn.neighbors, sklearn.preprocessing                                                     | synthetic_resampling, do_kde_binning, merge_single_sample_bins, identify_small_bins, merge_small_bins, calculate_centroid_distances, define_jenks_thresholds, calculate_bin_centers, assign_samples_to_bins, spatial_kde, define_density_thresholds, get_kde_bins, get_centroids, run_binned_smote, setup_synth_resampling, process_bins, custom_gpr_optimizer, cluster_minority_samples, __init__, _plot_cluster_weights, calculate_weights, _calculate_kmeans_weights, _calculate_kde_weights, _determine_eps, _calculate_dbscan_weights, _adjust_for_focus_regions, calculate_adaptive_bandwidth, find_optimal_clusters, __iter__, __len__                                                                                                                                                              | GeographicDensitySampler(__init__, _plot_cluster_weights, calculate_weights, _calculate_kmeans_weights, _calculate_kde_weights, _determine_eps, _calculate_dbscan_weights, _adjust_for_focus_regions, calculate_adaptive_bandwidth, find_optimal_clusters, __iter__, __len__)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |

## Module: __init__

### Imports

geogenie.geogenie

### Functions

None

### Classes

None

## Module: module_summary_script

### Imports

pathlib, pandas, graphviz

### Functions

parse_module_connections, extract_major_steps, extract_major_code_flow, save_major_steps_as_markdown, generate_major_code_flow_graph

### Classes

None

## Module: cli

### Imports

logging, pathlib, geogenie, geogenie.utils.argument_parser, geogenie.utils.logger

### Functions

main

### Classes

None

## Module: geogenie

### Imports

csv, json, logging, os, time, traceback, functools, pathlib, numpy, optuna, pandas, torch, torch.nn, torch.optim, xgboost, scipy.stats, sklearn.metrics, geogenie.models.models, geogenie.optimize.bootstrap, geogenie.optimize.optuna_opt, geogenie.plotting.plotting, geogenie.samplers.interpolate, geogenie.samplers.samplers, geogenie.utils.callbacks, geogenie.utils.data, geogenie.utils.data_structure, geogenie.utils.loss, geogenie.utils.scorers, geogenie.utils.utils

### Functions

timer, save_execution_times, wrapper, __init__, total_execution_time_decorator, load_data, save_model, train_rf, visualize_oversampling, train_model, train_step, test_step, _batch_init, compute_rolling_statistics, predict_locations, calculate_prediction_metrics, _create_metrics_dictionary, get_all_stats, print_stats_to_logger, get_correlation_coef, plot_bootstrap_aggregates, perform_standard_training, extract_best_params, write_pred_locations, load_best_params, optimize_parameters, perform_bootstrap_training, evaluate_and_save_results, make_unseen_predictions, train_test_predict, wrapper, rescale_predictions, rescale_predictions, mad, coefficient_of_variation, within_threshold

### Classes

GeoGenIE(__init__, total_execution_time_decorator, load_data, save_model, train_rf, visualize_oversampling, train_model, train_step, test_step, _batch_init, compute_rolling_statistics, predict_locations, calculate_prediction_metrics, _create_metrics_dictionary, get_all_stats, print_stats_to_logger, get_correlation_coef, plot_bootstrap_aggregates, perform_standard_training, extract_best_params, write_pred_locations, load_best_params, optimize_parameters, perform_bootstrap_training, evaluate_and_save_results, make_unseen_predictions, train_test_predict)

## Module: bootstrap

### Imports

json, logging, os, threading, concurrent.futures, copy, pathlib, numpy, pandas, torch, torch, torch.utils.data, geogenie.plotting.plotting, geogenie.samplers.interpolate, geogenie.utils.callbacks, geogenie.utils.data, geogenie.utils.exceptions, geogenie.utils.utils

### Functions

__init__, _get_thread_local_rng, _resample_loaders, _resample_boot, reset_weights, reinitialize_model, train_one_bootstrap, bootstrap_training_generator, extract_best_params, save_bootstrap_results, perform_bootstrap_training, _process_boot_preds, _grouped_ci_boot, _bootrep_metrics_to_csv, _validate_sample_data

### Classes

Bootstrap(__init__, _get_thread_local_rng, _resample_loaders, _resample_boot, reset_weights, reinitialize_model, train_one_bootstrap, bootstrap_training_generator, extract_best_params, save_bootstrap_results, perform_bootstrap_training, _process_boot_preds, _grouped_ci_boot, _bootrep_metrics_to_csv, _validate_sample_data)

## Module: optuna_opt

### Imports

json, logging, os, pickle, time, pathlib, numpy, optuna, pandas, torch, optuna, optuna.logging, torch, torch.utils.data, geogenie.plotting.plotting, geogenie.samplers.interpolate, geogenie.utils.callbacks, geogenie.utils.data, geogenie.utils.loss

### Functions

__init__, map_sampler_indices, objective_function, extract_features_labels, run_rf_training, run_training, set_gb_param_grid, set_param_grid, evaluate_model, perform_optuna_optimization, process_optuna_results, write_optuna_study_details, objective

### Classes

Optimize(__init__, map_sampler_indices, objective_function, extract_features_labels, run_rf_training, run_training, set_gb_param_grid, set_param_grid, evaluate_model, perform_optuna_optimization, process_optuna_results, write_optuna_study_details)

## Module: scorers

### Imports

logging, math, numba, numpy, scipy.stats, sklearn.manifold, sklearn.metrics, geogenie.utils.spatial_data_processors

### Functions

kstest, calculate_r2_knn, calculate_rmse, haversine_distance, predict, lle_reconstruction_scorer

### Classes

LocallyLinearEmbeddingWrapper(predict, lle_reconstruction_scorer)

## Module: transformers

### Imports

logging, numpy, scipy.sparse, sklearn.base, sklearn.preprocessing, sklearn.utils.extmath, sklearn.utils.validation

### Functions

__init__, fit, transform, _normalize_data, _compute_S_matrix, _store_results, __init__, fit, transform, inverse_transform

### Classes

MCA(__init__, fit, transform, _normalize_data, _compute_S_matrix, _store_results), MinMaxScalerGeo(__init__, fit, transform, inverse_transform)

## Module: __init__

### Imports

None

### Functions

None

### Classes

None

## Module: logger

### Imports

logging, pathlib

### Functions

setup_logger

### Classes

None

## Module: loss

### Imports

torch, torch.nn

### Functions

weighted_rmse_loss, __init__, forward, __init__, forward

### Classes

WeightedDRMSLoss(__init__, forward), WeightedHuberLoss(__init__, forward)

## Module: utils

### Imports

logging, signal, contextlib, io, numpy, pandas, torch, sklearn.cluster, geogenie.utils.exceptions

### Functions

check_column_dtype, detect_separator, read_csv_with_dynamic_sep, time_limit, validate_is_numpy, assign_to_bins, geo_coords_is_valid, get_iupac_dict, signal_handler

### Classes

None

## Module: callbacks

### Imports

logging, pathlib, numpy, torch

### Functions

callback_init, __init__, __call__, save_checkpoint, load_best_model

### Classes

EarlyStopping(__init__, __call__, save_checkpoint, load_best_model)

## Module: exceptions

### Imports

None

### Functions

__init__, __init__, __init__, __init__, __init__, __init__, __init__

### Classes

GPUUnavailableError(__init__), ResourceAllocationError(__init__), TimeoutException(None), DataStructureError(None), InvalidSampleDataError(__init__), SampleOrderingError(__init__), InvalidInputShapeError(__init__), EmbeddingError(__init__), OutlierDetectionError(__init__)

## Module: spatial_data_processors

### Imports

warnings, math, pathlib, geopandas, numpy, pandas, requests, geopy.distance, scipy.spatial, sklearn.cluster

### Functions

__init__, extract_basemap_path_url, to_pandas, to_numpy, to_geopandas, _ensure_is_pandas, _ensure_is_numpy, _ensure_is_gdf, haversine_distance, haversine_error, calculate_statistics, _validate_dists, spherical_mean, calculate_bounding_box, nearest_neighbor, calculate_convex_hull, detect_clusters, detect_outliers, geodesic_distance

### Classes

SpatialDataProcessor(__init__, extract_basemap_path_url, to_pandas, to_numpy, to_geopandas, _ensure_is_pandas, _ensure_is_numpy, _ensure_is_gdf, haversine_distance, haversine_error, calculate_statistics, _validate_dists, spherical_mean, calculate_bounding_box, nearest_neighbor, calculate_convex_hull, detect_clusters, detect_outliers, geodesic_distance)

## Module: data

### Imports

torch, torch.utils.data

### Functions

__init__, features, features, labels, labels, sample_weights, sample_weights, sample_ids, sample_ids, n_features, n_labels, __len__, __getitem__

### Classes

CustomDataset(__init__, features, features, labels, labels, sample_weights, sample_weights, sample_ids, sample_ids, n_features, n_labels, __len__, __getitem__)

## Module: data_structure

### Imports

logging, os, warnings, pathlib, numpy, pandas, pysam, torch, kneed, sklearn.base, sklearn.cluster, sklearn.decomposition, sklearn.impute, sklearn.manifold, sklearn.metrics, sklearn.model_selection, sklearn.neighbors, sklearn.preprocessing, torch.utils.data, geogenie.outliers.detect_outliers, geogenie.plotting.plotting, geogenie.samplers.samplers, geogenie.utils.data, geogenie.utils.exceptions, geogenie.utils.scorers, geogenie.utils.transformers, geogenie.utils.utils

### Functions

__init__, _load_vcf_file, _decompress_recompress_index_and_load_vcf, _decompress_vcf, _recompress_vcf, _index_vcf, _cleanup_decompressed_file, _parse_genotypes, map_alleles_to_iupac, is_biallelic, define_params, count_alleles, impute_missing, sort_samples, normalize_target, _check_sample_ordering, snps_to_012, filter_gt, _find_optimal_clusters, _determine_bandwidth, _determine_eps, _adjust_splits, split_train_test, map_outliers_through_filters, load_and_preprocess_data, generate_unknowns, extract_datasets, validate_feature_target_len, setup_index_masks, run_outlier_detection, call_create_dataloaders, embed, perform_mca_and_select_components, select_optimal_components, find_optimal_nmf_components, get_num_pca_comp, create_dataloaders, get_sample_weights, params, params

### Classes

DataStructure(__init__, _load_vcf_file, _decompress_recompress_index_and_load_vcf, _decompress_vcf, _recompress_vcf, _index_vcf, _cleanup_decompressed_file, _parse_genotypes, map_alleles_to_iupac, is_biallelic, define_params, count_alleles, impute_missing, sort_samples, normalize_target, _check_sample_ordering, snps_to_012, filter_gt, _find_optimal_clusters, _determine_bandwidth, _determine_eps, _adjust_splits, split_train_test, map_outliers_through_filters, load_and_preprocess_data, generate_unknowns, extract_datasets, validate_feature_target_len, setup_index_masks, run_outlier_detection, call_create_dataloaders, embed, perform_mca_and_select_components, select_optimal_components, find_optimal_nmf_components, get_num_pca_comp, create_dataloaders, get_sample_weights, params, params)

## Module: argument_parser

### Imports

argparse, ast, logging, os, warnings, yaml, torch.cuda, geogenie.utils.exceptions

### Functions

load_config, validate_positive_int, validate_positive_float, validate_gpu_number, validate_n_jobs, validate_split, validate_verbosity, validate_seed, validate_lower_str, setup_parser, validate_str2list, validate_dtype, validate_gb_params, validate_weighted_opts, validate_colorscale, validate_smote, validate_embeddings, validate_max_neighbors, validate_significance_levels, validate_inputs, __call__

### Classes

EvaluateAction(__call__)

## Module: models

### Imports

logging, numpy, torch, torch.nn

### Functions

__init__, _define_model, forward

### Classes

MLPRegressor(__init__, _define_model, forward)

## Module: __init__

### Imports

None

### Functions

None

### Classes

None

## Module: conf

### Imports

os, sys

### Functions

None

### Classes

None

## Module: detect_outliers

### Imports

logging, time, os, pathlib, numpy, pynndescent, scipy.optimize, scipy.spatial.distance, scipy.stats, geogenie.plotting.plotting, geogenie.utils.scorers, geogenie.utils.utils

### Functions

__init__, calculate_dgeo, calculate_statistic, rescale_statistic, find_gen_knn, find_geo_knn, find_optimal_k, predict_coords_knn, fit_gamma_mle, gamma_neg_log_likelihood, multi_stage_outlier_knn, analysis, run_multistage, filter_and_detect, search_nn_optk, plot_gamma_dist, detect_outliers, composite_outlier_detection

### Classes

GeoGeneticOutlierDetector(__init__, calculate_dgeo, calculate_statistic, rescale_statistic, find_gen_knn, find_geo_knn, find_optimal_k, predict_coords_knn, fit_gamma_mle, gamma_neg_log_likelihood, multi_stage_outlier_knn, analysis, run_multistage, filter_and_detect, search_nn_optk, plot_gamma_dist, detect_outliers, composite_outlier_detection)

## Module: __init__

### Imports

None

### Functions

None

### Classes

None

## Module: plotting

### Imports

logging, warnings, pathlib, geopandas, matplotlib, matplotlib.colors, matplotlib.lines, matplotlib.patches, matplotlib.pyplot, numpy, pandas, scipy.stats, seaborn, torch, kneed, optuna, optuna, pykrige.ok, scipy.stats, sklearn.exceptions, sklearn.linear_model, sklearn.pipeline, sklearn.preprocessing, geogenie.samplers.samplers, geogenie.utils.exceptions, geogenie.utils.spatial_data_processors, geogenie.utils.utils

### Functions

__init__, plot_times, plot_smote_bins, _remove_spines, _plot_smote_scatter, plot_history, make_optuna_plots, plot_bootstrap_aggregates, update_metric_labels, update_config_labels, plot_scatter_samples_map, plot_geographic_error_distribution, _run_kriging, _set_cbar_fontsize, _plot_scatter_map, _make_colorbar, plot_cumulative_error_distribution, _fill_kde_with_gradient, plot_zscores, plot_error_distribution, polynomial_regression_plot, plot_mca_curve, plot_nmf_error, plot_pca_curve, plot_outliers, plot_gamma_distribution, plot_sample_with_density, _highlight_counties, visualize_oversample_clusters, plot_data_distributions, pfx, pfx, outdir, outdir, obp, obp, roundup, roundup, roundup, calculate_95_ci, roundup

### Classes

PlotGenIE(__init__, plot_times, plot_smote_bins, _remove_spines, _plot_smote_scatter, plot_history, make_optuna_plots, plot_bootstrap_aggregates, update_metric_labels, update_config_labels, plot_scatter_samples_map, plot_geographic_error_distribution, _run_kriging, _set_cbar_fontsize, _plot_scatter_map, _make_colorbar, plot_cumulative_error_distribution, _fill_kde_with_gradient, plot_zscores, plot_error_distribution, polynomial_regression_plot, plot_mca_curve, plot_nmf_error, plot_pca_curve, plot_outliers, plot_gamma_distribution, plot_sample_with_density, _highlight_counties, visualize_oversample_clusters, plot_data_distributions, pfx, pfx, outdir, outdir, obp, obp)

## Module: __init__

### Imports

None

### Functions

None

### Classes

None

## Module: interpolate

### Imports

logging, copy, pathlib, matplotlib.pyplot, numpy, seaborn, torch, scipy.spatial.distance, sklearn.cluster, sklearn.metrics, sklearn.model_selection, sklearn.neighbors, torch.utils.data, geogenie.plotting.plotting, geogenie.utils.data

### Functions

run_genotype_interpolator, process_interp, resample_interp, reset_weighted_sampler, __init__, _determine_optimal_neighbors, _find_nearest_neighbors, interpolate_genotypes, _shuffle_over_sampled, _estimate_allele_frequencies, _vectorized_sample_hybrid, _assign_labels_to_synthetic_samples, _perform_kmeans_clustering, _calculate_centroids, _calculate_centroid_genotype, _calculate_optimal_bandwidth, _perform_density_estimation, _automated_parameter_tuning

### Classes

GenotypeInterpolator(__init__, _determine_optimal_neighbors, _find_nearest_neighbors, interpolate_genotypes, _shuffle_over_sampled, _estimate_allele_frequencies, _vectorized_sample_hybrid, _assign_labels_to_synthetic_samples, _perform_kmeans_clustering, _calculate_centroids, _calculate_centroid_genotype, _calculate_optimal_bandwidth, _perform_density_estimation, _automated_parameter_tuning)

## Module: samplers

### Imports

logging, os, jenkspy, matplotlib.pyplot, numpy, pandas, seaborn, torch, geopy.distance, imblearn.combine, imblearn.over_sampling, imblearn.under_sampling, scipy, scipy.spatial.distance, scipy.stats, sklearn.cluster, sklearn.metrics, sklearn.neighbors, sklearn.preprocessing, geogenie.utils.spatial_data_processors, geogenie.utils.utils, numpy, sklearn.cluster, sklearn.metrics, sklearn.neighbors, sklearn.preprocessing

### Functions

synthetic_resampling, do_kde_binning, merge_single_sample_bins, identify_small_bins, merge_small_bins, calculate_centroid_distances, define_jenks_thresholds, calculate_bin_centers, assign_samples_to_bins, spatial_kde, define_density_thresholds, get_kde_bins, get_centroids, run_binned_smote, setup_synth_resampling, process_bins, custom_gpr_optimizer, cluster_minority_samples, __init__, _plot_cluster_weights, calculate_weights, _calculate_kmeans_weights, _calculate_kde_weights, _determine_eps, _calculate_dbscan_weights, _adjust_for_focus_regions, calculate_adaptive_bandwidth, find_optimal_clusters, __iter__, __len__

### Classes

GeographicDensitySampler(__init__, _plot_cluster_weights, calculate_weights, _calculate_kmeans_weights, _calculate_kde_weights, _determine_eps, _calculate_dbscan_weights, _adjust_for_focus_regions, calculate_adaptive_bandwidth, find_optimal_clusters, __iter__, __len__)

