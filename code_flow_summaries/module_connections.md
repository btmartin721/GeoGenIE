# Module Relationships Summary

| Module                  | Type     | Name                                                     | Details                    |
|:------------------------|:---------|:---------------------------------------------------------|:---------------------------|
| __init__                | Import   | geogenie.geogenie                                        | Imported module            |
| cli                     | Import   | logging                                                  | Imported module            |
| cli                     | Import   | pathlib                                                  | Imported module            |
| cli                     | Import   | geogenie                                                 | Imported module            |
| cli                     | Import   | geogenie.utils.argument_parser                           | Imported module            |
| cli                     | Import   | geogenie.utils.logger                                    | Imported module            |
| cli                     | Function | main                                                     | Function defined in module |
| geogenie                | Import   | csv                                                      | Imported module            |
| geogenie                | Import   | json                                                     | Imported module            |
| geogenie                | Import   | logging                                                  | Imported module            |
| geogenie                | Import   | os                                                       | Imported module            |
| geogenie                | Import   | time                                                     | Imported module            |
| geogenie                | Import   | traceback                                                | Imported module            |
| geogenie                | Import   | functools                                                | Imported module            |
| geogenie                | Import   | pathlib                                                  | Imported module            |
| geogenie                | Import   | numpy                                                    | Imported module            |
| geogenie                | Import   | optuna                                                   | Imported module            |
| geogenie                | Import   | pandas                                                   | Imported module            |
| geogenie                | Import   | torch                                                    | Imported module            |
| geogenie                | Import   | torch.nn                                                 | Imported module            |
| geogenie                | Import   | torch.optim                                              | Imported module            |
| geogenie                | Import   | xgboost                                                  | Imported module            |
| geogenie                | Import   | scipy.stats                                              | Imported module            |
| geogenie                | Import   | sklearn.metrics                                          | Imported module            |
| geogenie                | Import   | geogenie.models.models                                   | Imported module            |
| geogenie                | Import   | geogenie.optimize.bootstrap                              | Imported module            |
| geogenie                | Import   | geogenie.optimize.optuna_opt                             | Imported module            |
| geogenie                | Import   | geogenie.plotting.plotting                               | Imported module            |
| geogenie                | Import   | geogenie.samplers.interpolate                            | Imported module            |
| geogenie                | Import   | geogenie.samplers.samplers                               | Imported module            |
| geogenie                | Import   | geogenie.utils.callbacks                                 | Imported module            |
| geogenie                | Import   | geogenie.utils.data                                      | Imported module            |
| geogenie                | Import   | geogenie.utils.data_structure                            | Imported module            |
| geogenie                | Import   | geogenie.utils.loss                                      | Imported module            |
| geogenie                | Import   | geogenie.utils.scorers                                   | Imported module            |
| geogenie                | Import   | geogenie.utils.utils                                     | Imported module            |
| geogenie                | Function | timer                                                    | Function defined in module |
| geogenie                | Function | save_execution_times                                     | Function defined in module |
| geogenie                | Function | wrapper                                                  | Function defined in module |
| geogenie                | Function | __init__                                                 | Function defined in module |
| geogenie                | Function | total_execution_time_decorator                           | Function defined in module |
| geogenie                | Function | load_data                                                | Function defined in module |
| geogenie                | Function | save_model                                               | Function defined in module |
| geogenie                | Function | train_rf                                                 | Function defined in module |
| geogenie                | Function | visualize_oversampling                                   | Function defined in module |
| geogenie                | Function | train_model                                              | Function defined in module |
| geogenie                | Function | train_step                                               | Function defined in module |
| geogenie                | Function | test_step                                                | Function defined in module |
| geogenie                | Function | _batch_init                                              | Function defined in module |
| geogenie                | Function | compute_rolling_statistics                               | Function defined in module |
| geogenie                | Function | predict_locations                                        | Function defined in module |
| geogenie                | Function | calculate_prediction_metrics                             | Function defined in module |
| geogenie                | Function | _create_metrics_dictionary                               | Function defined in module |
| geogenie                | Function | get_all_stats                                            | Function defined in module |
| geogenie                | Function | print_stats_to_logger                                    | Function defined in module |
| geogenie                | Function | get_correlation_coef                                     | Function defined in module |
| geogenie                | Function | plot_bootstrap_aggregates                                | Function defined in module |
| geogenie                | Function | perform_standard_training                                | Function defined in module |
| geogenie                | Function | extract_best_params                                      | Function defined in module |
| geogenie                | Function | write_pred_locations                                     | Function defined in module |
| geogenie                | Function | load_best_params                                         | Function defined in module |
| geogenie                | Function | optimize_parameters                                      | Function defined in module |
| geogenie                | Function | perform_bootstrap_training                               | Function defined in module |
| geogenie                | Function | evaluate_and_save_results                                | Function defined in module |
| geogenie                | Function | make_unseen_predictions                                  | Function defined in module |
| geogenie                | Function | train_test_predict                                       | Function defined in module |
| geogenie                | Function | wrapper                                                  | Function defined in module |
| geogenie                | Function | rescale_predictions                                      | Function defined in module |
| geogenie                | Function | rescale_predictions                                      | Function defined in module |
| geogenie                | Function | mad                                                      | Function defined in module |
| geogenie                | Function | coefficient_of_variation                                 | Function defined in module |
| geogenie                | Function | within_threshold                                         | Function defined in module |
| geogenie                | Class    | GeoGenIE                                                 | Class defined in module    |
| geogenie                | Method   | GeoGenIE.__init__                                        | Method defined in class    |
| geogenie                | Method   | GeoGenIE.total_execution_time_decorator                  | Method defined in class    |
| geogenie                | Method   | GeoGenIE.load_data                                       | Method defined in class    |
| geogenie                | Method   | GeoGenIE.save_model                                      | Method defined in class    |
| geogenie                | Method   | GeoGenIE.train_rf                                        | Method defined in class    |
| geogenie                | Method   | GeoGenIE.visualize_oversampling                          | Method defined in class    |
| geogenie                | Method   | GeoGenIE.train_model                                     | Method defined in class    |
| geogenie                | Method   | GeoGenIE.train_step                                      | Method defined in class    |
| geogenie                | Method   | GeoGenIE.test_step                                       | Method defined in class    |
| geogenie                | Method   | GeoGenIE._batch_init                                     | Method defined in class    |
| geogenie                | Method   | GeoGenIE.compute_rolling_statistics                      | Method defined in class    |
| geogenie                | Method   | GeoGenIE.predict_locations                               | Method defined in class    |
| geogenie                | Method   | GeoGenIE.calculate_prediction_metrics                    | Method defined in class    |
| geogenie                | Method   | GeoGenIE._create_metrics_dictionary                      | Method defined in class    |
| geogenie                | Method   | GeoGenIE.get_all_stats                                   | Method defined in class    |
| geogenie                | Method   | GeoGenIE.print_stats_to_logger                           | Method defined in class    |
| geogenie                | Method   | GeoGenIE.get_correlation_coef                            | Method defined in class    |
| geogenie                | Method   | GeoGenIE.plot_bootstrap_aggregates                       | Method defined in class    |
| geogenie                | Method   | GeoGenIE.perform_standard_training                       | Method defined in class    |
| geogenie                | Method   | GeoGenIE.extract_best_params                             | Method defined in class    |
| geogenie                | Method   | GeoGenIE.write_pred_locations                            | Method defined in class    |
| geogenie                | Method   | GeoGenIE.load_best_params                                | Method defined in class    |
| geogenie                | Method   | GeoGenIE.optimize_parameters                             | Method defined in class    |
| geogenie                | Method   | GeoGenIE.perform_bootstrap_training                      | Method defined in class    |
| geogenie                | Method   | GeoGenIE.evaluate_and_save_results                       | Method defined in class    |
| geogenie                | Method   | GeoGenIE.make_unseen_predictions                         | Method defined in class    |
| geogenie                | Method   | GeoGenIE.train_test_predict                              | Method defined in class    |
| bootstrap               | Import   | json                                                     | Imported module            |
| bootstrap               | Import   | logging                                                  | Imported module            |
| bootstrap               | Import   | os                                                       | Imported module            |
| bootstrap               | Import   | threading                                                | Imported module            |
| bootstrap               | Import   | concurrent.futures                                       | Imported module            |
| bootstrap               | Import   | copy                                                     | Imported module            |
| bootstrap               | Import   | pathlib                                                  | Imported module            |
| bootstrap               | Import   | numpy                                                    | Imported module            |
| bootstrap               | Import   | pandas                                                   | Imported module            |
| bootstrap               | Import   | torch                                                    | Imported module            |
| bootstrap               | Import   | torch                                                    | Imported module            |
| bootstrap               | Import   | torch.utils.data                                         | Imported module            |
| bootstrap               | Import   | geogenie.plotting.plotting                               | Imported module            |
| bootstrap               | Import   | geogenie.samplers.interpolate                            | Imported module            |
| bootstrap               | Import   | geogenie.utils.callbacks                                 | Imported module            |
| bootstrap               | Import   | geogenie.utils.data                                      | Imported module            |
| bootstrap               | Import   | geogenie.utils.exceptions                                | Imported module            |
| bootstrap               | Import   | geogenie.utils.utils                                     | Imported module            |
| bootstrap               | Function | __init__                                                 | Function defined in module |
| bootstrap               | Function | _get_thread_local_rng                                    | Function defined in module |
| bootstrap               | Function | _resample_loaders                                        | Function defined in module |
| bootstrap               | Function | _resample_boot                                           | Function defined in module |
| bootstrap               | Function | reset_weights                                            | Function defined in module |
| bootstrap               | Function | reinitialize_model                                       | Function defined in module |
| bootstrap               | Function | train_one_bootstrap                                      | Function defined in module |
| bootstrap               | Function | bootstrap_training_generator                             | Function defined in module |
| bootstrap               | Function | extract_best_params                                      | Function defined in module |
| bootstrap               | Function | save_bootstrap_results                                   | Function defined in module |
| bootstrap               | Function | perform_bootstrap_training                               | Function defined in module |
| bootstrap               | Function | _process_boot_preds                                      | Function defined in module |
| bootstrap               | Function | _grouped_ci_boot                                         | Function defined in module |
| bootstrap               | Function | _bootrep_metrics_to_csv                                  | Function defined in module |
| bootstrap               | Function | _validate_sample_data                                    | Function defined in module |
| bootstrap               | Class    | Bootstrap                                                | Class defined in module    |
| bootstrap               | Method   | Bootstrap.__init__                                       | Method defined in class    |
| bootstrap               | Method   | Bootstrap._get_thread_local_rng                          | Method defined in class    |
| bootstrap               | Method   | Bootstrap._resample_loaders                              | Method defined in class    |
| bootstrap               | Method   | Bootstrap._resample_boot                                 | Method defined in class    |
| bootstrap               | Method   | Bootstrap.reset_weights                                  | Method defined in class    |
| bootstrap               | Method   | Bootstrap.reinitialize_model                             | Method defined in class    |
| bootstrap               | Method   | Bootstrap.train_one_bootstrap                            | Method defined in class    |
| bootstrap               | Method   | Bootstrap.bootstrap_training_generator                   | Method defined in class    |
| bootstrap               | Method   | Bootstrap.extract_best_params                            | Method defined in class    |
| bootstrap               | Method   | Bootstrap.save_bootstrap_results                         | Method defined in class    |
| bootstrap               | Method   | Bootstrap.perform_bootstrap_training                     | Method defined in class    |
| bootstrap               | Method   | Bootstrap._process_boot_preds                            | Method defined in class    |
| bootstrap               | Method   | Bootstrap._grouped_ci_boot                               | Method defined in class    |
| bootstrap               | Method   | Bootstrap._bootrep_metrics_to_csv                        | Method defined in class    |
| bootstrap               | Method   | Bootstrap._validate_sample_data                          | Method defined in class    |
| optuna_opt              | Import   | json                                                     | Imported module            |
| optuna_opt              | Import   | logging                                                  | Imported module            |
| optuna_opt              | Import   | os                                                       | Imported module            |
| optuna_opt              | Import   | pickle                                                   | Imported module            |
| optuna_opt              | Import   | time                                                     | Imported module            |
| optuna_opt              | Import   | pathlib                                                  | Imported module            |
| optuna_opt              | Import   | numpy                                                    | Imported module            |
| optuna_opt              | Import   | optuna                                                   | Imported module            |
| optuna_opt              | Import   | pandas                                                   | Imported module            |
| optuna_opt              | Import   | torch                                                    | Imported module            |
| optuna_opt              | Import   | optuna                                                   | Imported module            |
| optuna_opt              | Import   | optuna.logging                                           | Imported module            |
| optuna_opt              | Import   | torch                                                    | Imported module            |
| optuna_opt              | Import   | torch.utils.data                                         | Imported module            |
| optuna_opt              | Import   | geogenie.plotting.plotting                               | Imported module            |
| optuna_opt              | Import   | geogenie.samplers.interpolate                            | Imported module            |
| optuna_opt              | Import   | geogenie.utils.callbacks                                 | Imported module            |
| optuna_opt              | Import   | geogenie.utils.data                                      | Imported module            |
| optuna_opt              | Import   | geogenie.utils.loss                                      | Imported module            |
| optuna_opt              | Function | __init__                                                 | Function defined in module |
| optuna_opt              | Function | map_sampler_indices                                      | Function defined in module |
| optuna_opt              | Function | objective_function                                       | Function defined in module |
| optuna_opt              | Function | extract_features_labels                                  | Function defined in module |
| optuna_opt              | Function | run_rf_training                                          | Function defined in module |
| optuna_opt              | Function | run_training                                             | Function defined in module |
| optuna_opt              | Function | set_gb_param_grid                                        | Function defined in module |
| optuna_opt              | Function | set_param_grid                                           | Function defined in module |
| optuna_opt              | Function | evaluate_model                                           | Function defined in module |
| optuna_opt              | Function | perform_optuna_optimization                              | Function defined in module |
| optuna_opt              | Function | process_optuna_results                                   | Function defined in module |
| optuna_opt              | Function | write_optuna_study_details                               | Function defined in module |
| optuna_opt              | Function | objective                                                | Function defined in module |
| optuna_opt              | Class    | Optimize                                                 | Class defined in module    |
| optuna_opt              | Method   | Optimize.__init__                                        | Method defined in class    |
| optuna_opt              | Method   | Optimize.map_sampler_indices                             | Method defined in class    |
| optuna_opt              | Method   | Optimize.objective_function                              | Method defined in class    |
| optuna_opt              | Method   | Optimize.extract_features_labels                         | Method defined in class    |
| optuna_opt              | Method   | Optimize.run_rf_training                                 | Method defined in class    |
| optuna_opt              | Method   | Optimize.run_training                                    | Method defined in class    |
| optuna_opt              | Method   | Optimize.set_gb_param_grid                               | Method defined in class    |
| optuna_opt              | Method   | Optimize.set_param_grid                                  | Method defined in class    |
| optuna_opt              | Method   | Optimize.evaluate_model                                  | Method defined in class    |
| optuna_opt              | Method   | Optimize.perform_optuna_optimization                     | Method defined in class    |
| optuna_opt              | Method   | Optimize.process_optuna_results                          | Method defined in class    |
| optuna_opt              | Method   | Optimize.write_optuna_study_details                      | Method defined in class    |
| scorers                 | Import   | logging                                                  | Imported module            |
| scorers                 | Import   | math                                                     | Imported module            |
| scorers                 | Import   | numba                                                    | Imported module            |
| scorers                 | Import   | numpy                                                    | Imported module            |
| scorers                 | Import   | scipy.stats                                              | Imported module            |
| scorers                 | Import   | sklearn.manifold                                         | Imported module            |
| scorers                 | Import   | sklearn.metrics                                          | Imported module            |
| scorers                 | Import   | geogenie.utils.spatial_data_processors                   | Imported module            |
| scorers                 | Function | kstest                                                   | Function defined in module |
| scorers                 | Function | calculate_r2_knn                                         | Function defined in module |
| scorers                 | Function | calculate_rmse                                           | Function defined in module |
| scorers                 | Function | haversine_distance                                       | Function defined in module |
| scorers                 | Function | predict                                                  | Function defined in module |
| scorers                 | Function | lle_reconstruction_scorer                                | Function defined in module |
| scorers                 | Class    | LocallyLinearEmbeddingWrapper                            | Class defined in module    |
| scorers                 | Method   | LocallyLinearEmbeddingWrapper.predict                    | Method defined in class    |
| scorers                 | Method   | LocallyLinearEmbeddingWrapper.lle_reconstruction_scorer  | Method defined in class    |
| transformers            | Import   | logging                                                  | Imported module            |
| transformers            | Import   | numpy                                                    | Imported module            |
| transformers            | Import   | scipy.sparse                                             | Imported module            |
| transformers            | Import   | sklearn.base                                             | Imported module            |
| transformers            | Import   | sklearn.preprocessing                                    | Imported module            |
| transformers            | Import   | sklearn.utils.extmath                                    | Imported module            |
| transformers            | Import   | sklearn.utils.validation                                 | Imported module            |
| transformers            | Function | __init__                                                 | Function defined in module |
| transformers            | Function | fit                                                      | Function defined in module |
| transformers            | Function | transform                                                | Function defined in module |
| transformers            | Function | _normalize_data                                          | Function defined in module |
| transformers            | Function | _compute_S_matrix                                        | Function defined in module |
| transformers            | Function | _store_results                                           | Function defined in module |
| transformers            | Function | __init__                                                 | Function defined in module |
| transformers            | Function | fit                                                      | Function defined in module |
| transformers            | Function | transform                                                | Function defined in module |
| transformers            | Function | inverse_transform                                        | Function defined in module |
| transformers            | Class    | MCA                                                      | Class defined in module    |
| transformers            | Method   | MCA.__init__                                             | Method defined in class    |
| transformers            | Method   | MCA.fit                                                  | Method defined in class    |
| transformers            | Method   | MCA.transform                                            | Method defined in class    |
| transformers            | Method   | MCA._normalize_data                                      | Method defined in class    |
| transformers            | Method   | MCA._compute_S_matrix                                    | Method defined in class    |
| transformers            | Method   | MCA._store_results                                       | Method defined in class    |
| transformers            | Class    | MinMaxScalerGeo                                          | Class defined in module    |
| transformers            | Method   | MinMaxScalerGeo.__init__                                 | Method defined in class    |
| transformers            | Method   | MinMaxScalerGeo.fit                                      | Method defined in class    |
| transformers            | Method   | MinMaxScalerGeo.transform                                | Method defined in class    |
| transformers            | Method   | MinMaxScalerGeo.inverse_transform                        | Method defined in class    |
| logger                  | Import   | logging                                                  | Imported module            |
| logger                  | Import   | pathlib                                                  | Imported module            |
| logger                  | Function | setup_logger                                             | Function defined in module |
| loss                    | Import   | torch                                                    | Imported module            |
| loss                    | Import   | torch.nn                                                 | Imported module            |
| loss                    | Function | weighted_rmse_loss                                       | Function defined in module |
| loss                    | Function | __init__                                                 | Function defined in module |
| loss                    | Function | forward                                                  | Function defined in module |
| loss                    | Function | __init__                                                 | Function defined in module |
| loss                    | Function | forward                                                  | Function defined in module |
| loss                    | Class    | WeightedDRMSLoss                                         | Class defined in module    |
| loss                    | Method   | WeightedDRMSLoss.__init__                                | Method defined in class    |
| loss                    | Method   | WeightedDRMSLoss.forward                                 | Method defined in class    |
| loss                    | Class    | WeightedHuberLoss                                        | Class defined in module    |
| loss                    | Method   | WeightedHuberLoss.__init__                               | Method defined in class    |
| loss                    | Method   | WeightedHuberLoss.forward                                | Method defined in class    |
| utils                   | Import   | logging                                                  | Imported module            |
| utils                   | Import   | signal                                                   | Imported module            |
| utils                   | Import   | contextlib                                               | Imported module            |
| utils                   | Import   | io                                                       | Imported module            |
| utils                   | Import   | numpy                                                    | Imported module            |
| utils                   | Import   | pandas                                                   | Imported module            |
| utils                   | Import   | torch                                                    | Imported module            |
| utils                   | Import   | sklearn.cluster                                          | Imported module            |
| utils                   | Import   | geogenie.utils.exceptions                                | Imported module            |
| utils                   | Function | check_column_dtype                                       | Function defined in module |
| utils                   | Function | detect_separator                                         | Function defined in module |
| utils                   | Function | read_csv_with_dynamic_sep                                | Function defined in module |
| utils                   | Function | time_limit                                               | Function defined in module |
| utils                   | Function | validate_is_numpy                                        | Function defined in module |
| utils                   | Function | assign_to_bins                                           | Function defined in module |
| utils                   | Function | geo_coords_is_valid                                      | Function defined in module |
| utils                   | Function | get_iupac_dict                                           | Function defined in module |
| utils                   | Function | signal_handler                                           | Function defined in module |
| callbacks               | Import   | logging                                                  | Imported module            |
| callbacks               | Import   | pathlib                                                  | Imported module            |
| callbacks               | Import   | numpy                                                    | Imported module            |
| callbacks               | Import   | torch                                                    | Imported module            |
| callbacks               | Function | callback_init                                            | Function defined in module |
| callbacks               | Function | __init__                                                 | Function defined in module |
| callbacks               | Function | __call__                                                 | Function defined in module |
| callbacks               | Function | save_checkpoint                                          | Function defined in module |
| callbacks               | Function | load_best_model                                          | Function defined in module |
| callbacks               | Class    | EarlyStopping                                            | Class defined in module    |
| callbacks               | Method   | EarlyStopping.__init__                                   | Method defined in class    |
| callbacks               | Method   | EarlyStopping.__call__                                   | Method defined in class    |
| callbacks               | Method   | EarlyStopping.save_checkpoint                            | Method defined in class    |
| callbacks               | Method   | EarlyStopping.load_best_model                            | Method defined in class    |
| exceptions              | Function | __init__                                                 | Function defined in module |
| exceptions              | Function | __init__                                                 | Function defined in module |
| exceptions              | Function | __init__                                                 | Function defined in module |
| exceptions              | Function | __init__                                                 | Function defined in module |
| exceptions              | Function | __init__                                                 | Function defined in module |
| exceptions              | Function | __init__                                                 | Function defined in module |
| exceptions              | Function | __init__                                                 | Function defined in module |
| exceptions              | Class    | GPUUnavailableError                                      | Class defined in module    |
| exceptions              | Method   | GPUUnavailableError.__init__                             | Method defined in class    |
| exceptions              | Class    | ResourceAllocationError                                  | Class defined in module    |
| exceptions              | Method   | ResourceAllocationError.__init__                         | Method defined in class    |
| exceptions              | Class    | TimeoutException                                         | Class defined in module    |
| exceptions              | Class    | DataStructureError                                       | Class defined in module    |
| exceptions              | Class    | InvalidSampleDataError                                   | Class defined in module    |
| exceptions              | Method   | InvalidSampleDataError.__init__                          | Method defined in class    |
| exceptions              | Class    | SampleOrderingError                                      | Class defined in module    |
| exceptions              | Method   | SampleOrderingError.__init__                             | Method defined in class    |
| exceptions              | Class    | InvalidInputShapeError                                   | Class defined in module    |
| exceptions              | Method   | InvalidInputShapeError.__init__                          | Method defined in class    |
| exceptions              | Class    | EmbeddingError                                           | Class defined in module    |
| exceptions              | Method   | EmbeddingError.__init__                                  | Method defined in class    |
| exceptions              | Class    | OutlierDetectionError                                    | Class defined in module    |
| exceptions              | Method   | OutlierDetectionError.__init__                           | Method defined in class    |
| spatial_data_processors | Import   | warnings                                                 | Imported module            |
| spatial_data_processors | Import   | math                                                     | Imported module            |
| spatial_data_processors | Import   | pathlib                                                  | Imported module            |
| spatial_data_processors | Import   | geopandas                                                | Imported module            |
| spatial_data_processors | Import   | numpy                                                    | Imported module            |
| spatial_data_processors | Import   | pandas                                                   | Imported module            |
| spatial_data_processors | Import   | requests                                                 | Imported module            |
| spatial_data_processors | Import   | geopy.distance                                           | Imported module            |
| spatial_data_processors | Import   | scipy.spatial                                            | Imported module            |
| spatial_data_processors | Import   | sklearn.cluster                                          | Imported module            |
| spatial_data_processors | Function | __init__                                                 | Function defined in module |
| spatial_data_processors | Function | extract_basemap_path_url                                 | Function defined in module |
| spatial_data_processors | Function | to_pandas                                                | Function defined in module |
| spatial_data_processors | Function | to_numpy                                                 | Function defined in module |
| spatial_data_processors | Function | to_geopandas                                             | Function defined in module |
| spatial_data_processors | Function | _ensure_is_pandas                                        | Function defined in module |
| spatial_data_processors | Function | _ensure_is_numpy                                         | Function defined in module |
| spatial_data_processors | Function | _ensure_is_gdf                                           | Function defined in module |
| spatial_data_processors | Function | haversine_distance                                       | Function defined in module |
| spatial_data_processors | Function | haversine_error                                          | Function defined in module |
| spatial_data_processors | Function | calculate_statistics                                     | Function defined in module |
| spatial_data_processors | Function | _validate_dists                                          | Function defined in module |
| spatial_data_processors | Function | spherical_mean                                           | Function defined in module |
| spatial_data_processors | Function | calculate_bounding_box                                   | Function defined in module |
| spatial_data_processors | Function | nearest_neighbor                                         | Function defined in module |
| spatial_data_processors | Function | calculate_convex_hull                                    | Function defined in module |
| spatial_data_processors | Function | detect_clusters                                          | Function defined in module |
| spatial_data_processors | Function | detect_outliers                                          | Function defined in module |
| spatial_data_processors | Function | geodesic_distance                                        | Function defined in module |
| spatial_data_processors | Class    | SpatialDataProcessor                                     | Class defined in module    |
| spatial_data_processors | Method   | SpatialDataProcessor.__init__                            | Method defined in class    |
| spatial_data_processors | Method   | SpatialDataProcessor.extract_basemap_path_url            | Method defined in class    |
| spatial_data_processors | Method   | SpatialDataProcessor.to_pandas                           | Method defined in class    |
| spatial_data_processors | Method   | SpatialDataProcessor.to_numpy                            | Method defined in class    |
| spatial_data_processors | Method   | SpatialDataProcessor.to_geopandas                        | Method defined in class    |
| spatial_data_processors | Method   | SpatialDataProcessor._ensure_is_pandas                   | Method defined in class    |
| spatial_data_processors | Method   | SpatialDataProcessor._ensure_is_numpy                    | Method defined in class    |
| spatial_data_processors | Method   | SpatialDataProcessor._ensure_is_gdf                      | Method defined in class    |
| spatial_data_processors | Method   | SpatialDataProcessor.haversine_distance                  | Method defined in class    |
| spatial_data_processors | Method   | SpatialDataProcessor.haversine_error                     | Method defined in class    |
| spatial_data_processors | Method   | SpatialDataProcessor.calculate_statistics                | Method defined in class    |
| spatial_data_processors | Method   | SpatialDataProcessor._validate_dists                     | Method defined in class    |
| spatial_data_processors | Method   | SpatialDataProcessor.spherical_mean                      | Method defined in class    |
| spatial_data_processors | Method   | SpatialDataProcessor.calculate_bounding_box              | Method defined in class    |
| spatial_data_processors | Method   | SpatialDataProcessor.nearest_neighbor                    | Method defined in class    |
| spatial_data_processors | Method   | SpatialDataProcessor.calculate_convex_hull               | Method defined in class    |
| spatial_data_processors | Method   | SpatialDataProcessor.detect_clusters                     | Method defined in class    |
| spatial_data_processors | Method   | SpatialDataProcessor.detect_outliers                     | Method defined in class    |
| spatial_data_processors | Method   | SpatialDataProcessor.geodesic_distance                   | Method defined in class    |
| data                    | Import   | torch                                                    | Imported module            |
| data                    | Import   | torch.utils.data                                         | Imported module            |
| data                    | Function | __init__                                                 | Function defined in module |
| data                    | Function | features                                                 | Function defined in module |
| data                    | Function | features                                                 | Function defined in module |
| data                    | Function | labels                                                   | Function defined in module |
| data                    | Function | labels                                                   | Function defined in module |
| data                    | Function | sample_weights                                           | Function defined in module |
| data                    | Function | sample_weights                                           | Function defined in module |
| data                    | Function | sample_ids                                               | Function defined in module |
| data                    | Function | sample_ids                                               | Function defined in module |
| data                    | Function | n_features                                               | Function defined in module |
| data                    | Function | n_labels                                                 | Function defined in module |
| data                    | Function | __len__                                                  | Function defined in module |
| data                    | Function | __getitem__                                              | Function defined in module |
| data                    | Class    | CustomDataset                                            | Class defined in module    |
| data                    | Method   | CustomDataset.__init__                                   | Method defined in class    |
| data                    | Method   | CustomDataset.features                                   | Method defined in class    |
| data                    | Method   | CustomDataset.features                                   | Method defined in class    |
| data                    | Method   | CustomDataset.labels                                     | Method defined in class    |
| data                    | Method   | CustomDataset.labels                                     | Method defined in class    |
| data                    | Method   | CustomDataset.sample_weights                             | Method defined in class    |
| data                    | Method   | CustomDataset.sample_weights                             | Method defined in class    |
| data                    | Method   | CustomDataset.sample_ids                                 | Method defined in class    |
| data                    | Method   | CustomDataset.sample_ids                                 | Method defined in class    |
| data                    | Method   | CustomDataset.n_features                                 | Method defined in class    |
| data                    | Method   | CustomDataset.n_labels                                   | Method defined in class    |
| data                    | Method   | CustomDataset.__len__                                    | Method defined in class    |
| data                    | Method   | CustomDataset.__getitem__                                | Method defined in class    |
| data_structure          | Import   | logging                                                  | Imported module            |
| data_structure          | Import   | os                                                       | Imported module            |
| data_structure          | Import   | warnings                                                 | Imported module            |
| data_structure          | Import   | pathlib                                                  | Imported module            |
| data_structure          | Import   | numpy                                                    | Imported module            |
| data_structure          | Import   | pandas                                                   | Imported module            |
| data_structure          | Import   | pysam                                                    | Imported module            |
| data_structure          | Import   | torch                                                    | Imported module            |
| data_structure          | Import   | kneed                                                    | Imported module            |
| data_structure          | Import   | sklearn.base                                             | Imported module            |
| data_structure          | Import   | sklearn.cluster                                          | Imported module            |
| data_structure          | Import   | sklearn.decomposition                                    | Imported module            |
| data_structure          | Import   | sklearn.impute                                           | Imported module            |
| data_structure          | Import   | sklearn.manifold                                         | Imported module            |
| data_structure          | Import   | sklearn.metrics                                          | Imported module            |
| data_structure          | Import   | sklearn.model_selection                                  | Imported module            |
| data_structure          | Import   | sklearn.neighbors                                        | Imported module            |
| data_structure          | Import   | sklearn.preprocessing                                    | Imported module            |
| data_structure          | Import   | torch.utils.data                                         | Imported module            |
| data_structure          | Import   | geogenie.outliers.detect_outliers                        | Imported module            |
| data_structure          | Import   | geogenie.plotting.plotting                               | Imported module            |
| data_structure          | Import   | geogenie.samplers.samplers                               | Imported module            |
| data_structure          | Import   | geogenie.utils.data                                      | Imported module            |
| data_structure          | Import   | geogenie.utils.exceptions                                | Imported module            |
| data_structure          | Import   | geogenie.utils.scorers                                   | Imported module            |
| data_structure          | Import   | geogenie.utils.transformers                              | Imported module            |
| data_structure          | Import   | geogenie.utils.utils                                     | Imported module            |
| data_structure          | Function | __init__                                                 | Function defined in module |
| data_structure          | Function | _load_vcf_file                                           | Function defined in module |
| data_structure          | Function | _decompress_recompress_index_and_load_vcf                | Function defined in module |
| data_structure          | Function | _decompress_vcf                                          | Function defined in module |
| data_structure          | Function | _recompress_vcf                                          | Function defined in module |
| data_structure          | Function | _index_vcf                                               | Function defined in module |
| data_structure          | Function | _cleanup_decompressed_file                               | Function defined in module |
| data_structure          | Function | _parse_genotypes                                         | Function defined in module |
| data_structure          | Function | map_alleles_to_iupac                                     | Function defined in module |
| data_structure          | Function | is_biallelic                                             | Function defined in module |
| data_structure          | Function | define_params                                            | Function defined in module |
| data_structure          | Function | count_alleles                                            | Function defined in module |
| data_structure          | Function | impute_missing                                           | Function defined in module |
| data_structure          | Function | sort_samples                                             | Function defined in module |
| data_structure          | Function | normalize_target                                         | Function defined in module |
| data_structure          | Function | _check_sample_ordering                                   | Function defined in module |
| data_structure          | Function | snps_to_012                                              | Function defined in module |
| data_structure          | Function | filter_gt                                                | Function defined in module |
| data_structure          | Function | _find_optimal_clusters                                   | Function defined in module |
| data_structure          | Function | _determine_bandwidth                                     | Function defined in module |
| data_structure          | Function | _determine_eps                                           | Function defined in module |
| data_structure          | Function | _adjust_splits                                           | Function defined in module |
| data_structure          | Function | split_train_test                                         | Function defined in module |
| data_structure          | Function | map_outliers_through_filters                             | Function defined in module |
| data_structure          | Function | load_and_preprocess_data                                 | Function defined in module |
| data_structure          | Function | generate_unknowns                                        | Function defined in module |
| data_structure          | Function | extract_datasets                                         | Function defined in module |
| data_structure          | Function | validate_feature_target_len                              | Function defined in module |
| data_structure          | Function | setup_index_masks                                        | Function defined in module |
| data_structure          | Function | run_outlier_detection                                    | Function defined in module |
| data_structure          | Function | call_create_dataloaders                                  | Function defined in module |
| data_structure          | Function | embed                                                    | Function defined in module |
| data_structure          | Function | perform_mca_and_select_components                        | Function defined in module |
| data_structure          | Function | select_optimal_components                                | Function defined in module |
| data_structure          | Function | find_optimal_nmf_components                              | Function defined in module |
| data_structure          | Function | get_num_pca_comp                                         | Function defined in module |
| data_structure          | Function | create_dataloaders                                       | Function defined in module |
| data_structure          | Function | get_sample_weights                                       | Function defined in module |
| data_structure          | Function | params                                                   | Function defined in module |
| data_structure          | Function | params                                                   | Function defined in module |
| data_structure          | Class    | DataStructure                                            | Class defined in module    |
| data_structure          | Method   | DataStructure.__init__                                   | Method defined in class    |
| data_structure          | Method   | DataStructure._load_vcf_file                             | Method defined in class    |
| data_structure          | Method   | DataStructure._decompress_recompress_index_and_load_vcf  | Method defined in class    |
| data_structure          | Method   | DataStructure._decompress_vcf                            | Method defined in class    |
| data_structure          | Method   | DataStructure._recompress_vcf                            | Method defined in class    |
| data_structure          | Method   | DataStructure._index_vcf                                 | Method defined in class    |
| data_structure          | Method   | DataStructure._cleanup_decompressed_file                 | Method defined in class    |
| data_structure          | Method   | DataStructure._parse_genotypes                           | Method defined in class    |
| data_structure          | Method   | DataStructure.map_alleles_to_iupac                       | Method defined in class    |
| data_structure          | Method   | DataStructure.is_biallelic                               | Method defined in class    |
| data_structure          | Method   | DataStructure.define_params                              | Method defined in class    |
| data_structure          | Method   | DataStructure.count_alleles                              | Method defined in class    |
| data_structure          | Method   | DataStructure.impute_missing                             | Method defined in class    |
| data_structure          | Method   | DataStructure.sort_samples                               | Method defined in class    |
| data_structure          | Method   | DataStructure.normalize_target                           | Method defined in class    |
| data_structure          | Method   | DataStructure._check_sample_ordering                     | Method defined in class    |
| data_structure          | Method   | DataStructure.snps_to_012                                | Method defined in class    |
| data_structure          | Method   | DataStructure.filter_gt                                  | Method defined in class    |
| data_structure          | Method   | DataStructure._find_optimal_clusters                     | Method defined in class    |
| data_structure          | Method   | DataStructure._determine_bandwidth                       | Method defined in class    |
| data_structure          | Method   | DataStructure._determine_eps                             | Method defined in class    |
| data_structure          | Method   | DataStructure._adjust_splits                             | Method defined in class    |
| data_structure          | Method   | DataStructure.split_train_test                           | Method defined in class    |
| data_structure          | Method   | DataStructure.map_outliers_through_filters               | Method defined in class    |
| data_structure          | Method   | DataStructure.load_and_preprocess_data                   | Method defined in class    |
| data_structure          | Method   | DataStructure.generate_unknowns                          | Method defined in class    |
| data_structure          | Method   | DataStructure.extract_datasets                           | Method defined in class    |
| data_structure          | Method   | DataStructure.validate_feature_target_len                | Method defined in class    |
| data_structure          | Method   | DataStructure.setup_index_masks                          | Method defined in class    |
| data_structure          | Method   | DataStructure.run_outlier_detection                      | Method defined in class    |
| data_structure          | Method   | DataStructure.call_create_dataloaders                    | Method defined in class    |
| data_structure          | Method   | DataStructure.embed                                      | Method defined in class    |
| data_structure          | Method   | DataStructure.perform_mca_and_select_components          | Method defined in class    |
| data_structure          | Method   | DataStructure.select_optimal_components                  | Method defined in class    |
| data_structure          | Method   | DataStructure.find_optimal_nmf_components                | Method defined in class    |
| data_structure          | Method   | DataStructure.get_num_pca_comp                           | Method defined in class    |
| data_structure          | Method   | DataStructure.create_dataloaders                         | Method defined in class    |
| data_structure          | Method   | DataStructure.get_sample_weights                         | Method defined in class    |
| data_structure          | Method   | DataStructure.params                                     | Method defined in class    |
| data_structure          | Method   | DataStructure.params                                     | Method defined in class    |
| argument_parser         | Import   | argparse                                                 | Imported module            |
| argument_parser         | Import   | ast                                                      | Imported module            |
| argument_parser         | Import   | logging                                                  | Imported module            |
| argument_parser         | Import   | os                                                       | Imported module            |
| argument_parser         | Import   | warnings                                                 | Imported module            |
| argument_parser         | Import   | yaml                                                     | Imported module            |
| argument_parser         | Import   | torch.cuda                                               | Imported module            |
| argument_parser         | Import   | geogenie.utils.exceptions                                | Imported module            |
| argument_parser         | Function | load_config                                              | Function defined in module |
| argument_parser         | Function | validate_positive_int                                    | Function defined in module |
| argument_parser         | Function | validate_positive_float                                  | Function defined in module |
| argument_parser         | Function | validate_gpu_number                                      | Function defined in module |
| argument_parser         | Function | validate_n_jobs                                          | Function defined in module |
| argument_parser         | Function | validate_split                                           | Function defined in module |
| argument_parser         | Function | validate_verbosity                                       | Function defined in module |
| argument_parser         | Function | validate_seed                                            | Function defined in module |
| argument_parser         | Function | validate_lower_str                                       | Function defined in module |
| argument_parser         | Function | setup_parser                                             | Function defined in module |
| argument_parser         | Function | validate_str2list                                        | Function defined in module |
| argument_parser         | Function | validate_dtype                                           | Function defined in module |
| argument_parser         | Function | validate_gb_params                                       | Function defined in module |
| argument_parser         | Function | validate_weighted_opts                                   | Function defined in module |
| argument_parser         | Function | validate_colorscale                                      | Function defined in module |
| argument_parser         | Function | validate_smote                                           | Function defined in module |
| argument_parser         | Function | validate_embeddings                                      | Function defined in module |
| argument_parser         | Function | validate_max_neighbors                                   | Function defined in module |
| argument_parser         | Function | validate_significance_levels                             | Function defined in module |
| argument_parser         | Function | validate_inputs                                          | Function defined in module |
| argument_parser         | Function | __call__                                                 | Function defined in module |
| argument_parser         | Class    | EvaluateAction                                           | Class defined in module    |
| argument_parser         | Method   | EvaluateAction.__call__                                  | Method defined in class    |
| models                  | Import   | logging                                                  | Imported module            |
| models                  | Import   | numpy                                                    | Imported module            |
| models                  | Import   | torch                                                    | Imported module            |
| models                  | Import   | torch.nn                                                 | Imported module            |
| models                  | Function | __init__                                                 | Function defined in module |
| models                  | Function | _define_model                                            | Function defined in module |
| models                  | Function | forward                                                  | Function defined in module |
| models                  | Class    | MLPRegressor                                             | Class defined in module    |
| models                  | Method   | MLPRegressor.__init__                                    | Method defined in class    |
| models                  | Method   | MLPRegressor._define_model                               | Method defined in class    |
| models                  | Method   | MLPRegressor.forward                                     | Method defined in class    |
| conf                    | Import   | os                                                       | Imported module            |
| conf                    | Import   | sys                                                      | Imported module            |
| detect_outliers         | Import   | logging                                                  | Imported module            |
| detect_outliers         | Import   | time                                                     | Imported module            |
| detect_outliers         | Import   | os                                                       | Imported module            |
| detect_outliers         | Import   | pathlib                                                  | Imported module            |
| detect_outliers         | Import   | numpy                                                    | Imported module            |
| detect_outliers         | Import   | pynndescent                                              | Imported module            |
| detect_outliers         | Import   | scipy.optimize                                           | Imported module            |
| detect_outliers         | Import   | scipy.spatial.distance                                   | Imported module            |
| detect_outliers         | Import   | scipy.stats                                              | Imported module            |
| detect_outliers         | Import   | geogenie.plotting.plotting                               | Imported module            |
| detect_outliers         | Import   | geogenie.utils.scorers                                   | Imported module            |
| detect_outliers         | Import   | geogenie.utils.utils                                     | Imported module            |
| detect_outliers         | Function | __init__                                                 | Function defined in module |
| detect_outliers         | Function | calculate_dgeo                                           | Function defined in module |
| detect_outliers         | Function | calculate_statistic                                      | Function defined in module |
| detect_outliers         | Function | rescale_statistic                                        | Function defined in module |
| detect_outliers         | Function | find_gen_knn                                             | Function defined in module |
| detect_outliers         | Function | find_geo_knn                                             | Function defined in module |
| detect_outliers         | Function | find_optimal_k                                           | Function defined in module |
| detect_outliers         | Function | predict_coords_knn                                       | Function defined in module |
| detect_outliers         | Function | fit_gamma_mle                                            | Function defined in module |
| detect_outliers         | Function | gamma_neg_log_likelihood                                 | Function defined in module |
| detect_outliers         | Function | multi_stage_outlier_knn                                  | Function defined in module |
| detect_outliers         | Function | analysis                                                 | Function defined in module |
| detect_outliers         | Function | run_multistage                                           | Function defined in module |
| detect_outliers         | Function | filter_and_detect                                        | Function defined in module |
| detect_outliers         | Function | search_nn_optk                                           | Function defined in module |
| detect_outliers         | Function | plot_gamma_dist                                          | Function defined in module |
| detect_outliers         | Function | detect_outliers                                          | Function defined in module |
| detect_outliers         | Function | composite_outlier_detection                              | Function defined in module |
| detect_outliers         | Class    | GeoGeneticOutlierDetector                                | Class defined in module    |
| detect_outliers         | Method   | GeoGeneticOutlierDetector.__init__                       | Method defined in class    |
| detect_outliers         | Method   | GeoGeneticOutlierDetector.calculate_dgeo                 | Method defined in class    |
| detect_outliers         | Method   | GeoGeneticOutlierDetector.calculate_statistic            | Method defined in class    |
| detect_outliers         | Method   | GeoGeneticOutlierDetector.rescale_statistic              | Method defined in class    |
| detect_outliers         | Method   | GeoGeneticOutlierDetector.find_gen_knn                   | Method defined in class    |
| detect_outliers         | Method   | GeoGeneticOutlierDetector.find_geo_knn                   | Method defined in class    |
| detect_outliers         | Method   | GeoGeneticOutlierDetector.find_optimal_k                 | Method defined in class    |
| detect_outliers         | Method   | GeoGeneticOutlierDetector.predict_coords_knn             | Method defined in class    |
| detect_outliers         | Method   | GeoGeneticOutlierDetector.fit_gamma_mle                  | Method defined in class    |
| detect_outliers         | Method   | GeoGeneticOutlierDetector.gamma_neg_log_likelihood       | Method defined in class    |
| detect_outliers         | Method   | GeoGeneticOutlierDetector.multi_stage_outlier_knn        | Method defined in class    |
| detect_outliers         | Method   | GeoGeneticOutlierDetector.analysis                       | Method defined in class    |
| detect_outliers         | Method   | GeoGeneticOutlierDetector.run_multistage                 | Method defined in class    |
| detect_outliers         | Method   | GeoGeneticOutlierDetector.filter_and_detect              | Method defined in class    |
| detect_outliers         | Method   | GeoGeneticOutlierDetector.search_nn_optk                 | Method defined in class    |
| detect_outliers         | Method   | GeoGeneticOutlierDetector.plot_gamma_dist                | Method defined in class    |
| detect_outliers         | Method   | GeoGeneticOutlierDetector.detect_outliers                | Method defined in class    |
| detect_outliers         | Method   | GeoGeneticOutlierDetector.composite_outlier_detection    | Method defined in class    |
| plotting                | Import   | logging                                                  | Imported module            |
| plotting                | Import   | warnings                                                 | Imported module            |
| plotting                | Import   | pathlib                                                  | Imported module            |
| plotting                | Import   | geopandas                                                | Imported module            |
| plotting                | Import   | matplotlib                                               | Imported module            |
| plotting                | Import   | matplotlib.colors                                        | Imported module            |
| plotting                | Import   | matplotlib.lines                                         | Imported module            |
| plotting                | Import   | matplotlib.patches                                       | Imported module            |
| plotting                | Import   | matplotlib.pyplot                                        | Imported module            |
| plotting                | Import   | numpy                                                    | Imported module            |
| plotting                | Import   | pandas                                                   | Imported module            |
| plotting                | Import   | scipy.stats                                              | Imported module            |
| plotting                | Import   | seaborn                                                  | Imported module            |
| plotting                | Import   | torch                                                    | Imported module            |
| plotting                | Import   | kneed                                                    | Imported module            |
| plotting                | Import   | optuna                                                   | Imported module            |
| plotting                | Import   | optuna                                                   | Imported module            |
| plotting                | Import   | pykrige.ok                                               | Imported module            |
| plotting                | Import   | scipy.stats                                              | Imported module            |
| plotting                | Import   | sklearn.exceptions                                       | Imported module            |
| plotting                | Import   | sklearn.linear_model                                     | Imported module            |
| plotting                | Import   | sklearn.pipeline                                         | Imported module            |
| plotting                | Import   | sklearn.preprocessing                                    | Imported module            |
| plotting                | Import   | geogenie.samplers.samplers                               | Imported module            |
| plotting                | Import   | geogenie.utils.exceptions                                | Imported module            |
| plotting                | Import   | geogenie.utils.spatial_data_processors                   | Imported module            |
| plotting                | Import   | geogenie.utils.utils                                     | Imported module            |
| plotting                | Function | __init__                                                 | Function defined in module |
| plotting                | Function | plot_times                                               | Function defined in module |
| plotting                | Function | plot_smote_bins                                          | Function defined in module |
| plotting                | Function | _remove_spines                                           | Function defined in module |
| plotting                | Function | _plot_smote_scatter                                      | Function defined in module |
| plotting                | Function | plot_history                                             | Function defined in module |
| plotting                | Function | make_optuna_plots                                        | Function defined in module |
| plotting                | Function | plot_bootstrap_aggregates                                | Function defined in module |
| plotting                | Function | update_metric_labels                                     | Function defined in module |
| plotting                | Function | update_config_labels                                     | Function defined in module |
| plotting                | Function | plot_scatter_samples_map                                 | Function defined in module |
| plotting                | Function | plot_geographic_error_distribution                       | Function defined in module |
| plotting                | Function | _run_kriging                                             | Function defined in module |
| plotting                | Function | _set_cbar_fontsize                                       | Function defined in module |
| plotting                | Function | _plot_scatter_map                                        | Function defined in module |
| plotting                | Function | _make_colorbar                                           | Function defined in module |
| plotting                | Function | plot_cumulative_error_distribution                       | Function defined in module |
| plotting                | Function | _fill_kde_with_gradient                                  | Function defined in module |
| plotting                | Function | plot_zscores                                             | Function defined in module |
| plotting                | Function | plot_error_distribution                                  | Function defined in module |
| plotting                | Function | polynomial_regression_plot                               | Function defined in module |
| plotting                | Function | plot_mca_curve                                           | Function defined in module |
| plotting                | Function | plot_nmf_error                                           | Function defined in module |
| plotting                | Function | plot_pca_curve                                           | Function defined in module |
| plotting                | Function | plot_outliers                                            | Function defined in module |
| plotting                | Function | plot_gamma_distribution                                  | Function defined in module |
| plotting                | Function | plot_sample_with_density                                 | Function defined in module |
| plotting                | Function | _highlight_counties                                      | Function defined in module |
| plotting                | Function | visualize_oversample_clusters                            | Function defined in module |
| plotting                | Function | plot_data_distributions                                  | Function defined in module |
| plotting                | Function | pfx                                                      | Function defined in module |
| plotting                | Function | pfx                                                      | Function defined in module |
| plotting                | Function | outdir                                                   | Function defined in module |
| plotting                | Function | outdir                                                   | Function defined in module |
| plotting                | Function | obp                                                      | Function defined in module |
| plotting                | Function | obp                                                      | Function defined in module |
| plotting                | Function | roundup                                                  | Function defined in module |
| plotting                | Function | roundup                                                  | Function defined in module |
| plotting                | Function | roundup                                                  | Function defined in module |
| plotting                | Function | calculate_95_ci                                          | Function defined in module |
| plotting                | Function | roundup                                                  | Function defined in module |
| plotting                | Class    | PlotGenIE                                                | Class defined in module    |
| plotting                | Method   | PlotGenIE.__init__                                       | Method defined in class    |
| plotting                | Method   | PlotGenIE.plot_times                                     | Method defined in class    |
| plotting                | Method   | PlotGenIE.plot_smote_bins                                | Method defined in class    |
| plotting                | Method   | PlotGenIE._remove_spines                                 | Method defined in class    |
| plotting                | Method   | PlotGenIE._plot_smote_scatter                            | Method defined in class    |
| plotting                | Method   | PlotGenIE.plot_history                                   | Method defined in class    |
| plotting                | Method   | PlotGenIE.make_optuna_plots                              | Method defined in class    |
| plotting                | Method   | PlotGenIE.plot_bootstrap_aggregates                      | Method defined in class    |
| plotting                | Method   | PlotGenIE.update_metric_labels                           | Method defined in class    |
| plotting                | Method   | PlotGenIE.update_config_labels                           | Method defined in class    |
| plotting                | Method   | PlotGenIE.plot_scatter_samples_map                       | Method defined in class    |
| plotting                | Method   | PlotGenIE.plot_geographic_error_distribution             | Method defined in class    |
| plotting                | Method   | PlotGenIE._run_kriging                                   | Method defined in class    |
| plotting                | Method   | PlotGenIE._set_cbar_fontsize                             | Method defined in class    |
| plotting                | Method   | PlotGenIE._plot_scatter_map                              | Method defined in class    |
| plotting                | Method   | PlotGenIE._make_colorbar                                 | Method defined in class    |
| plotting                | Method   | PlotGenIE.plot_cumulative_error_distribution             | Method defined in class    |
| plotting                | Method   | PlotGenIE._fill_kde_with_gradient                        | Method defined in class    |
| plotting                | Method   | PlotGenIE.plot_zscores                                   | Method defined in class    |
| plotting                | Method   | PlotGenIE.plot_error_distribution                        | Method defined in class    |
| plotting                | Method   | PlotGenIE.polynomial_regression_plot                     | Method defined in class    |
| plotting                | Method   | PlotGenIE.plot_mca_curve                                 | Method defined in class    |
| plotting                | Method   | PlotGenIE.plot_nmf_error                                 | Method defined in class    |
| plotting                | Method   | PlotGenIE.plot_pca_curve                                 | Method defined in class    |
| plotting                | Method   | PlotGenIE.plot_outliers                                  | Method defined in class    |
| plotting                | Method   | PlotGenIE.plot_gamma_distribution                        | Method defined in class    |
| plotting                | Method   | PlotGenIE.plot_sample_with_density                       | Method defined in class    |
| plotting                | Method   | PlotGenIE._highlight_counties                            | Method defined in class    |
| plotting                | Method   | PlotGenIE.visualize_oversample_clusters                  | Method defined in class    |
| plotting                | Method   | PlotGenIE.plot_data_distributions                        | Method defined in class    |
| plotting                | Method   | PlotGenIE.pfx                                            | Method defined in class    |
| plotting                | Method   | PlotGenIE.pfx                                            | Method defined in class    |
| plotting                | Method   | PlotGenIE.outdir                                         | Method defined in class    |
| plotting                | Method   | PlotGenIE.outdir                                         | Method defined in class    |
| plotting                | Method   | PlotGenIE.obp                                            | Method defined in class    |
| plotting                | Method   | PlotGenIE.obp                                            | Method defined in class    |
| interpolate             | Import   | logging                                                  | Imported module            |
| interpolate             | Import   | copy                                                     | Imported module            |
| interpolate             | Import   | pathlib                                                  | Imported module            |
| interpolate             | Import   | matplotlib.pyplot                                        | Imported module            |
| interpolate             | Import   | numpy                                                    | Imported module            |
| interpolate             | Import   | seaborn                                                  | Imported module            |
| interpolate             | Import   | torch                                                    | Imported module            |
| interpolate             | Import   | scipy.spatial.distance                                   | Imported module            |
| interpolate             | Import   | sklearn.cluster                                          | Imported module            |
| interpolate             | Import   | sklearn.metrics                                          | Imported module            |
| interpolate             | Import   | sklearn.model_selection                                  | Imported module            |
| interpolate             | Import   | sklearn.neighbors                                        | Imported module            |
| interpolate             | Import   | torch.utils.data                                         | Imported module            |
| interpolate             | Import   | geogenie.plotting.plotting                               | Imported module            |
| interpolate             | Import   | geogenie.utils.data                                      | Imported module            |
| interpolate             | Function | run_genotype_interpolator                                | Function defined in module |
| interpolate             | Function | process_interp                                           | Function defined in module |
| interpolate             | Function | resample_interp                                          | Function defined in module |
| interpolate             | Function | reset_weighted_sampler                                   | Function defined in module |
| interpolate             | Function | __init__                                                 | Function defined in module |
| interpolate             | Function | _determine_optimal_neighbors                             | Function defined in module |
| interpolate             | Function | _find_nearest_neighbors                                  | Function defined in module |
| interpolate             | Function | interpolate_genotypes                                    | Function defined in module |
| interpolate             | Function | _shuffle_over_sampled                                    | Function defined in module |
| interpolate             | Function | _estimate_allele_frequencies                             | Function defined in module |
| interpolate             | Function | _vectorized_sample_hybrid                                | Function defined in module |
| interpolate             | Function | _assign_labels_to_synthetic_samples                      | Function defined in module |
| interpolate             | Function | _perform_kmeans_clustering                               | Function defined in module |
| interpolate             | Function | _calculate_centroids                                     | Function defined in module |
| interpolate             | Function | _calculate_centroid_genotype                             | Function defined in module |
| interpolate             | Function | _calculate_optimal_bandwidth                             | Function defined in module |
| interpolate             | Function | _perform_density_estimation                              | Function defined in module |
| interpolate             | Function | _automated_parameter_tuning                              | Function defined in module |
| interpolate             | Class    | GenotypeInterpolator                                     | Class defined in module    |
| interpolate             | Method   | GenotypeInterpolator.__init__                            | Method defined in class    |
| interpolate             | Method   | GenotypeInterpolator._determine_optimal_neighbors        | Method defined in class    |
| interpolate             | Method   | GenotypeInterpolator._find_nearest_neighbors             | Method defined in class    |
| interpolate             | Method   | GenotypeInterpolator.interpolate_genotypes               | Method defined in class    |
| interpolate             | Method   | GenotypeInterpolator._shuffle_over_sampled               | Method defined in class    |
| interpolate             | Method   | GenotypeInterpolator._estimate_allele_frequencies        | Method defined in class    |
| interpolate             | Method   | GenotypeInterpolator._vectorized_sample_hybrid           | Method defined in class    |
| interpolate             | Method   | GenotypeInterpolator._assign_labels_to_synthetic_samples | Method defined in class    |
| interpolate             | Method   | GenotypeInterpolator._perform_kmeans_clustering          | Method defined in class    |
| interpolate             | Method   | GenotypeInterpolator._calculate_centroids                | Method defined in class    |
| interpolate             | Method   | GenotypeInterpolator._calculate_centroid_genotype        | Method defined in class    |
| interpolate             | Method   | GenotypeInterpolator._calculate_optimal_bandwidth        | Method defined in class    |
| interpolate             | Method   | GenotypeInterpolator._perform_density_estimation         | Method defined in class    |
| interpolate             | Method   | GenotypeInterpolator._automated_parameter_tuning         | Method defined in class    |
| samplers                | Import   | logging                                                  | Imported module            |
| samplers                | Import   | os                                                       | Imported module            |
| samplers                | Import   | jenkspy                                                  | Imported module            |
| samplers                | Import   | matplotlib.pyplot                                        | Imported module            |
| samplers                | Import   | numpy                                                    | Imported module            |
| samplers                | Import   | pandas                                                   | Imported module            |
| samplers                | Import   | seaborn                                                  | Imported module            |
| samplers                | Import   | torch                                                    | Imported module            |
| samplers                | Import   | geopy.distance                                           | Imported module            |
| samplers                | Import   | imblearn.combine                                         | Imported module            |
| samplers                | Import   | imblearn.over_sampling                                   | Imported module            |
| samplers                | Import   | imblearn.under_sampling                                  | Imported module            |
| samplers                | Import   | scipy                                                    | Imported module            |
| samplers                | Import   | scipy.spatial.distance                                   | Imported module            |
| samplers                | Import   | scipy.stats                                              | Imported module            |
| samplers                | Import   | sklearn.cluster                                          | Imported module            |
| samplers                | Import   | sklearn.metrics                                          | Imported module            |
| samplers                | Import   | sklearn.neighbors                                        | Imported module            |
| samplers                | Import   | sklearn.preprocessing                                    | Imported module            |
| samplers                | Import   | geogenie.utils.spatial_data_processors                   | Imported module            |
| samplers                | Import   | geogenie.utils.utils                                     | Imported module            |
| samplers                | Import   | numpy                                                    | Imported module            |
| samplers                | Import   | sklearn.cluster                                          | Imported module            |
| samplers                | Import   | sklearn.metrics                                          | Imported module            |
| samplers                | Import   | sklearn.neighbors                                        | Imported module            |
| samplers                | Import   | sklearn.preprocessing                                    | Imported module            |
| samplers                | Function | synthetic_resampling                                     | Function defined in module |
| samplers                | Function | do_kde_binning                                           | Function defined in module |
| samplers                | Function | merge_single_sample_bins                                 | Function defined in module |
| samplers                | Function | identify_small_bins                                      | Function defined in module |
| samplers                | Function | merge_small_bins                                         | Function defined in module |
| samplers                | Function | calculate_centroid_distances                             | Function defined in module |
| samplers                | Function | define_jenks_thresholds                                  | Function defined in module |
| samplers                | Function | calculate_bin_centers                                    | Function defined in module |
| samplers                | Function | assign_samples_to_bins                                   | Function defined in module |
| samplers                | Function | spatial_kde                                              | Function defined in module |
| samplers                | Function | define_density_thresholds                                | Function defined in module |
| samplers                | Function | get_kde_bins                                             | Function defined in module |
| samplers                | Function | get_centroids                                            | Function defined in module |
| samplers                | Function | run_binned_smote                                         | Function defined in module |
| samplers                | Function | setup_synth_resampling                                   | Function defined in module |
| samplers                | Function | process_bins                                             | Function defined in module |
| samplers                | Function | custom_gpr_optimizer                                     | Function defined in module |
| samplers                | Function | cluster_minority_samples                                 | Function defined in module |
| samplers                | Function | __init__                                                 | Function defined in module |
| samplers                | Function | _plot_cluster_weights                                    | Function defined in module |
| samplers                | Function | calculate_weights                                        | Function defined in module |
| samplers                | Function | _calculate_kmeans_weights                                | Function defined in module |
| samplers                | Function | _calculate_kde_weights                                   | Function defined in module |
| samplers                | Function | _determine_eps                                           | Function defined in module |
| samplers                | Function | _calculate_dbscan_weights                                | Function defined in module |
| samplers                | Function | _adjust_for_focus_regions                                | Function defined in module |
| samplers                | Function | calculate_adaptive_bandwidth                             | Function defined in module |
| samplers                | Function | find_optimal_clusters                                    | Function defined in module |
| samplers                | Function | __iter__                                                 | Function defined in module |
| samplers                | Function | __len__                                                  | Function defined in module |
| samplers                | Class    | GeographicDensitySampler                                 | Class defined in module    |
| samplers                | Method   | GeographicDensitySampler.__init__                        | Method defined in class    |
| samplers                | Method   | GeographicDensitySampler._plot_cluster_weights           | Method defined in class    |
| samplers                | Method   | GeographicDensitySampler.calculate_weights               | Method defined in class    |
| samplers                | Method   | GeographicDensitySampler._calculate_kmeans_weights       | Method defined in class    |
| samplers                | Method   | GeographicDensitySampler._calculate_kde_weights          | Method defined in class    |
| samplers                | Method   | GeographicDensitySampler._determine_eps                  | Method defined in class    |
| samplers                | Method   | GeographicDensitySampler._calculate_dbscan_weights       | Method defined in class    |
| samplers                | Method   | GeographicDensitySampler._adjust_for_focus_regions       | Method defined in class    |
| samplers                | Method   | GeographicDensitySampler.calculate_adaptive_bandwidth    | Method defined in class    |
| samplers                | Method   | GeographicDensitySampler.find_optimal_clusters           | Method defined in class    |
| samplers                | Method   | GeographicDensitySampler.__iter__                        | Method defined in class    |
| samplers                | Method   | GeographicDensitySampler.__len__                         | Method defined in class    |