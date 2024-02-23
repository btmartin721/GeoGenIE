#!/bin/zsh

# Define the options for criterion and boolean values
criterion_options=("drms" "huber" "rmse")
boolean_options=("true" "false")

# Iterate over each combination of options
for criterion in $criterion_options; do
    for use_kmeans in $boolean_options; do
        for use_kde in $boolean_options; do
            # Construct the output directory name based on the current combination of options
            output_dir="params_comparison/${criterion}_${use_kmeans}_kmeans_${use_kde}_kde"

            # Create a temporary config file with the replacements
            sed -e "s|@|$output_dir|g" \
                -e "s|xxxx|$criterion|g" \
                -e "s|&|$use_kmeans|g" \
                -e "s|%|$use_kde|g" \
                config_files/config_testing.yaml > config_files/temp_config.yaml

            # Run the script with the modified config file
            python scripts/run_geogenie.py --config config_files/temp_config.yaml
        done
    done
done

# Remove the temporary config file after the script runs
rm config_files/temp_config.yaml
