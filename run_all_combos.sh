#!/bin/bash

# Define the arrays of options
embedding_types=("mca")
use_weighteds=("loss" "none" "sampler")
oversample_methods=("kerneldensity")
gb_use_lr_schedulers=("true")
force_weighted_opts=("true" "false")
force_no_weightings=("true" "false")
detect_outliers_opts=("true")
use_synthetic_oversamplings=("true" "false")
use_kmeans_opts=("false")
use_kde_opts=("true")
use_gradient_boosting_opts=("true" "false")

# Directory to save the generated scripts
mkdir -p "./generated_scripts_reduced"

# Generate scripts for each combination
for embedding_type in "${embedding_types[@]}"; do
    for use_weighted in "${use_weighteds[@]}"; do
        for oversample_method in "${oversample_methods[@]}"; do
            for gb_use_lr_scheduler in "${gb_use_lr_schedulers[@]}"; do
                for force_weighted_opt in "${force_weighted_opts[@]}"; do
                    for force_no_weighting in "${force_no_weightings[@]}"; do
                        for detect_outliers in "${detect_outliers_opts[@]}"; do
                            for use_synthetic_oversampling in "${use_synthetic_oversamplings[@]}"; do
                                for use_kmeans in "${use_kmeans_opts[@]}"; do
                                    for use_kde in "${use_kde_opts[@]}"; do
                                        for use_gradient_boosting in "${use_gradient_boosting_opts[@]}"; do
                                            # Construct script filename
                                            script_name="./generated_scripts_reduced/run_${embedding_type}_${use_weighted}_${oversample_method}_${gb_use_lr_scheduler}_${force_weighted_opt}_${force_no_weighting}_${detect_outliers}_${use_synthetic_oversampling}_${use_kmeans}_${use_kde}_${use_gradient_boosting}.sh"

                                            # Construct prefix and sqldb path
                                            prefix="config_${embedding_type}_${use_weighted}_${oversample_method}_${gb_use_lr_scheduler}_${force_weighted_opt}_${force_no_weighting}_${detect_outliers}_${use_synthetic_oversampling}_${use_kmeans}_${use_kde}_${use_gradient_boosting}"

                                            sqldb_path="./final_analysis/${prefix}/database"

                                            # Start writing the script
                                            echo "#!/bin/bash" > "$script_name"
                                            echo "python ./scripts/run_geogenie.py \\" >> "$script_name"
                                            echo "    --embedding_type $embedding_type \\" >> "$script_name"
                                            echo "    --use_weighted $use_weighted \\" >> "$script_name"
                                            echo "    --oversample_method $oversample_method \\" >> "$script_name"

                                            # Add toggle parameters only if they are true
                                            [[ $gb_use_lr_scheduler == "true" ]] && echo "    --gb_use_lr_scheduler \\" >> "$script_name"
                                            [[ $force_weighted_opt == "true" ]] && echo "    --force_weighted_opt \\" >> "$script_name"
                                            [[ $force_no_weighting == "true" ]] && echo "    --force_no_weighting \\" >> "$script_name"
                                            [[ $detect_outliers == "true" ]] && echo "    --detect_outliers \\" >> "$script_name"
                                            [[ $use_synthetic_oversampling == "true" ]] && echo "    --use_synthetic_oversampling \\" >> "$script_name"
                                            [[ $use_kmeans == "true" ]] && echo "    --use_kmeans \\" >> "$script_name"
                                            [[ $use_kde == "true" ]] && echo "    --use_kde \\" >> "$script_name"

                                            [[ $use_gradient_boosting == "true" ]] && echo "    --use_gradient_boosting \\" >> "$script_name"

                                            # Fixed option
                                            echo "    --verbose 0 \\" >> "$script_name"
                                            echo "    --vcf data/phase6_gtseq_subset.vcf.gz \\" >> "$script_name"
                                            echo "    --sample_data data/wtd_coords_N1426.txt \\" >> "$script_name"

                                            echo "    --n_iter 200 \\" >> "$script_name"

                                            echo "    --n_jobs 32 \\" >> "$script_name"

                                            echo "    --do_gridsearch \\" >> "$script_name"

                                            echo "    --prefix $prefix \\" >> "$script_name"

                                            echo "    --sqldb $sqldb_path \\" >> "$script_name"

                                            echo "    --output_dir final_analysis \\" >> "$script_name"

                                            # Make the script executable
                                            chmod +x "$script_name";
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "Generated scripts in ./generated_scripts_reduced"
