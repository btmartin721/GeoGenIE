#!/bin/zsh

# Define arrays for each parameter
kmeans="false"
kde="true"
crit="rmse"
oversample_methods=("none" "kmeans")
use_weighteds=("loss" "none")
detect_outliers=("true" "false")

train_split=0.75
val_split=0.25
lr_factor=0.5
factor=1.0
bin=6
batchsize=64
vcf="/Users/btm002/Documents/research/GeoGenIE/data/wtd_N1149_N436.vcf.gz"
sampledata="/Users/btm002/Documents/research/GeoGenIE/data/wtd_coords_N1149.csv"
n_jobs=8
counties="'Benton,Washington,Scott,Crawford,Washington,Sebastian,Yell,Logan,Franklin,Madison,Carroll,Boone,Newton,Johnson,Pope,Van Buren,Searcy,Marion,Baxter,Stone,Independence,Jackson,Randolph,Bradley,Union,Ashley'"


# Directory to store outputs
output_dir="/Users/btm002/Documents/research/GeoGenIE/analyses/wtd_gtseq_N1149_L436_sweep_28Nov2024"
mkdir -p $output_dir

# Function to generate commands
generate_commands() {

    # Initialize counter
    counter=1

    # Get the number of combinations
    n_combos=8

    for method in $oversample_methods; do
    for weighted in $use_weighteds; do
    for outlier in $detect_outliers; do
        if [[ $counter -le 2 ]]; then
            echo "Skipping command $counter / ${n_combos}..."
            counter=$((counter+1))
            continue
        fi

        prefix=oversample_${method}_weighted_${weighted}_outlier_${outlier}
        command="geogenie --oversample_method $method --train_split $train_split --val_split $val_split --lr_scheduler_factor $lr_factor --factor $factor --use_weighted $weighted --n_bins $bin --do_gridsearch --n_iter 100 --do_bootstrap --nboots 100 --output_dir $output_dir --prefix $prefix --batch_size $batchsize --criterion $crit --vcf $vcf --sample_data $sampledata --known_sample_data $sampledata --n_jobs $n_jobs --basemap_fips 05 --bbox_buffer 0.1 --load_best_params /Users/btm002/Documents/research/GeoGenIE/analyses/wtd_gg_27Nov2024_best/optimize/wtd_gg_27Nov2024_best_N1149_L436_best_params.json --highlight_basemap_counties $counties --verbose 0 --seed 42 --normalize_sample_weights --filetype pdf --remove_splines --fontsize 24 --max_neighbors 3 --maxk 50 --max_clusters 20"

        [[ $kmeans == "true" ]] && command+=" --use_kmeans"
        [[ $kde == "true" ]] && command+=" --use_kde"
        [[ $outlier == "true" ]] && command+=" --detect_outliers"

        echo "---------------------------------------------"
        echo "Running command: $prefix"
        echo "Executing command $counter / ${n_combos}..."
        echo "% Complete: $((counter * 100 / n_combos))%"
        echo "---------------------------------------------"

        # Execute the generated command
        eval $command 

        # Increment counter
        counter=$((counter+1))

    done;
    done;
    done;
}

# Run commands sequentially
generate_commands

echo "All training combinations have been executed. Parameter sweep complete!"
