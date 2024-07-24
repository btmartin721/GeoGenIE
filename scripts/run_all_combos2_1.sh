#!/bin/zsh

# Define arrays for each parameter
embedding_types=("none")
use_kmeans=("true")
use_kde=("false")
oversample_methods=("none" "kmeans")
criterion=("rmse")
train_splits=(0.7)
lr_scheduler_factors=(0.5)
factors=(1.0)
use_weighteds=("loss" "none")
n_bins=(5)
detect_outliers=("true" "false")
min_nn_dists=(1000)
scale_factors=(100)
batch_size=(64)
w_power=(1.0)
min_mac=(2)

# Directory to store outputs
output_dir="/Users/btm002/Documents/wtd/GeoGenIE/analyses/wtd_gtseq_N1415_L436_sweep_18Jul24"
mkdir -p $output_dir

# Function to generate commands
generate_commands() {
    for embedding in $embedding_types; do
        for kmeans in $use_kmeans; do
            for kde in $use_kde; do
                for method in $oversample_methods; do
                    for train_split in $train_splits; do
                        val_split=$(echo "scale=2; (1 - $train_split) / 2" | bc)
                        for lr_factor in $lr_scheduler_factors; do
                            for factor in $factors; do
                                for weighted in $use_weighteds; do
                                    for bin in $n_bins; do
                                        for mac in $min_mac; do
                                            for batchsize in $batch_size; do
                                                for crit in $criterion; do
                                                    for wp in $w_power; do
                                                        for outlier in $detect_outliers; do
                                                            for nn_dist in $min_nn_dists; do
                                                                for scale_factor in $scale_factors; do
                                                                    command="python scripts/run_geogenie.py --embedding_type $embedding --oversample_method $method --train_split $train_split --val_split $val_split --lr_scheduler_factor $lr_factor --factor $factor --use_weighted $weighted --n_bins $bin --min_nn_dist $nn_dist --scale_factor $scale_factor --do_bootstrap --nboots 100 --output_dir $output_dir --prefix emb_${embedding}_meth_${method}_tr_${train_split}_vr_${val_split}_lrf_${lr_factor}_fac_${factor}_wt_${weighted}_bins_${bin}_ndist_${nn_dist}_scl_${scale_factor}_km_${kmeans}_kd_${kde}_out_${outlier}_bs_${batchsize}_wp_${wp}_crit_${crit}_mac_${mac} --batch_size $batchsize --criterion $crit --w_power $wp --vcf /Users/btm002/Documents/wtd/GeoGenIE/data/final_analysis/with_outliers/wtd_gtseq_genotypes1415inds436snps.vcf.gz --sample_data /Users/btm002/Documents/wtd/GeoGenIE/data/wtd_coords_N1426.txt --n_jobs 8 --known_sample_data /Users/btm002/Documents/wtd/GeoGenIE/data/final_analysis/wtd_gtseq_coords1415inds.csv --basemap_fips 05 --bbox_buffer 0.1 --highlight_basemap_counties 'Benton,Washington,Scott,Crawford,Washington,Sebastian,Yell,Logan,Franklin,Madison,Carroll,Boone,Newton,Johnson,Pope,Van Buren,Searcy,Marion,Baxter,Stone,Independence,Jackson,Randolph,Bradley,Union,Ashley'"

                                                                    [[ $kmeans == "true" ]] && command+=" --use_kmeans"
                                                                    [[ $kde == "true" ]] && command+=" --use_kde"
                                                                    [[ $outlier == "true" ]] && command+=" --detect_outliers"
                                                                    echo $command
                                                                    eval $command # Execute the generated command
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
                    done
                done
            done
        done
    done
}

# Run commands sequentially
generate_commands

echo "All training combinations have been executed."
