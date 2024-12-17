# Major Code Flow Steps

| Module/Method          | Purpose                                | Dependencies              |
|------------------------|----------------------------------------|---------------------------|
| geogenie.train_test_predict | Main entry point for training, testing, and predictions. | Calls major downstream methods. |
| data_structure.sklearn.preprocessing | Performs key operations in data_structure (e.g., preprocessing, bootstrap training). | Imported module |
| data_structure.split_train_test | Performs key operations in data_structure (e.g., preprocessing, bootstrap training). | Function defined in module |
| data_structure.load_and_preprocess_data | Performs key operations in data_structure (e.g., preprocessing, bootstrap training). | Function defined in module |
| data_structure.DataStructure.split_train_test | Performs key operations in data_structure (e.g., preprocessing, bootstrap training). | Method defined in class |
| data_structure.DataStructure.load_and_preprocess_data | Performs key operations in data_structure (e.g., preprocessing, bootstrap training). | Method defined in class |
| bootstrap.train_one_bootstrap | Performs key operations in bootstrap (e.g., preprocessing, bootstrap training). | Function defined in module |
| bootstrap.bootstrap_training_generator | Performs key operations in bootstrap (e.g., preprocessing, bootstrap training). | Function defined in module |
| bootstrap.save_bootstrap_results | Performs key operations in bootstrap (e.g., preprocessing, bootstrap training). | Function defined in module |
| bootstrap.perform_bootstrap_training | Performs key operations in bootstrap (e.g., preprocessing, bootstrap training). | Function defined in module |
| bootstrap._process_boot_preds | Performs key operations in bootstrap (e.g., preprocessing, bootstrap training). | Function defined in module |
| bootstrap.Bootstrap | Performs key operations in bootstrap (e.g., preprocessing, bootstrap training). | Class defined in module |
| bootstrap.Bootstrap.__init__ | Performs key operations in bootstrap (e.g., preprocessing, bootstrap training). | Method defined in class |
| bootstrap.Bootstrap._get_thread_local_rng | Performs key operations in bootstrap (e.g., preprocessing, bootstrap training). | Method defined in class |
| bootstrap.Bootstrap._resample_loaders | Performs key operations in bootstrap (e.g., preprocessing, bootstrap training). | Method defined in class |
| bootstrap.Bootstrap._resample_boot | Performs key operations in bootstrap (e.g., preprocessing, bootstrap training). | Method defined in class |
| bootstrap.Bootstrap.reset_weights | Performs key operations in bootstrap (e.g., preprocessing, bootstrap training). | Method defined in class |
| bootstrap.Bootstrap.reinitialize_model | Performs key operations in bootstrap (e.g., preprocessing, bootstrap training). | Method defined in class |
| bootstrap.Bootstrap.train_one_bootstrap | Performs key operations in bootstrap (e.g., preprocessing, bootstrap training). | Method defined in class |
| bootstrap.Bootstrap.bootstrap_training_generator | Performs key operations in bootstrap (e.g., preprocessing, bootstrap training). | Method defined in class |
| bootstrap.Bootstrap.extract_best_params | Performs key operations in bootstrap (e.g., preprocessing, bootstrap training). | Method defined in class |
| bootstrap.Bootstrap.save_bootstrap_results | Performs key operations in bootstrap (e.g., preprocessing, bootstrap training). | Method defined in class |
| bootstrap.Bootstrap.perform_bootstrap_training | Performs key operations in bootstrap (e.g., preprocessing, bootstrap training). | Method defined in class |
| bootstrap.Bootstrap._process_boot_preds | Performs key operations in bootstrap (e.g., preprocessing, bootstrap training). | Method defined in class |
| bootstrap.Bootstrap._grouped_ci_boot | Performs key operations in bootstrap (e.g., preprocessing, bootstrap training). | Method defined in class |
| bootstrap.Bootstrap._bootrep_metrics_to_csv | Performs key operations in bootstrap (e.g., preprocessing, bootstrap training). | Method defined in class |
| bootstrap.Bootstrap._validate_sample_data | Performs key operations in bootstrap (e.g., preprocessing, bootstrap training). | Method defined in class |
| optuna_opt.run_rf_training | Performs key operations in optuna_opt (e.g., preprocessing, bootstrap training). | Function defined in module |
| optuna_opt.run_training | Performs key operations in optuna_opt (e.g., preprocessing, bootstrap training). | Function defined in module |
| optuna_opt.process_optuna_results | Performs key operations in optuna_opt (e.g., preprocessing, bootstrap training). | Function defined in module |
| optuna_opt.Optimize.run_rf_training | Performs key operations in optuna_opt (e.g., preprocessing, bootstrap training). | Method defined in class |
| optuna_opt.Optimize.run_training | Performs key operations in optuna_opt (e.g., preprocessing, bootstrap training). | Method defined in class |
| optuna_opt.Optimize.process_optuna_results | Performs key operations in optuna_opt (e.g., preprocessing, bootstrap training). | Method defined in class |
| interpolate.process_interp | Performs key operations in interpolate (e.g., preprocessing, bootstrap training). | Function defined in module |
| samplers.sklearn.preprocessing | Performs key operations in samplers (e.g., preprocessing, bootstrap training). | Imported module |
| samplers.geogenie.utils.spatial_data_processors | Performs key operations in samplers (e.g., preprocessing, bootstrap training). | Imported module |
| samplers.sklearn.preprocessing | Performs key operations in samplers (e.g., preprocessing, bootstrap training). | Imported module |
| samplers.process_bins | Performs key operations in samplers (e.g., preprocessing, bootstrap training). | Function defined in module |
