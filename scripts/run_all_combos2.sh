#!/bin/zsh

# Define arrays for each parameter
embedding_types=("none")
use_kmeans=("true")
use_kde=("false")
oversample_methods=("kmeans")
criterion=("rmse")
train_splits=(0.8)
lr_scheduler_factors=(0.75)
factors=(0.9)
use_weighteds=("loss" "sampler" "none")
n_bins=(5)
detect_outliers=("true" "false")
min_nn_dists=(1000)
scale_factors=(100)
batch_size=(64)
w_power=(1.0)
min_mac=(3)

# Directory to store outputs
output_dir="/Users/btm002/Documents/wtd/GeoGenIE/analyses/all_model_outputs_final_really_really3"
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
                                                                    command="python scripts/run_geogenie.py --embedding_type $embedding --oversample_method $method --train_split $train_split --val_split $val_split --lr_scheduler_factor $lr_factor --factor $factor --use_weighted $weighted --n_bins $bin --min_nn_dist $nn_dist --scale_factor $scale_factor --do_bootstrap --nboots 100 --output_dir $output_dir --prefix emb_${embedding}_meth_${method}_tr_${train_split}_vr_${val_split}_lrf_${lr_factor}_fac_${factor}_wt_${weighted}_bins_${bin}_ndist_${nn_dist}_scl_${scale_factor}_km_${kmeans}_kd_${kde}_out_${outlier}_bs_${batchsize}_wp_${wp}_crit_${crit}_mac_${mac} --batch_size $batchsize --criterion $crit --w_power $wp --vcf /Users/btm002/Documents/wtd/GeoGenIE/data/sorted_merged_gtseq_radseq.vcf.gz --sample_data /Users/btm002/Documents/wtd/GeoGenIE/data/wtd_coords_N1426.txt --n_jobs 8 --known_sample_data /Users/btm002/Documents/wtd/GeoGenIE/data/wtd_fy23_samples_known.txt --basemap_fips 05 --bbox_buffer 0.2 --samples_to_plot '83NW1N010,83NW1P048,83NW2N209,83NW1N001,83HM2N001,83FR2N017,83FR2N015,83FR2N011,83FR2N014,83PP1N010,83PP1N040,83PP1N034,83PP1N045,83WD2N001,83WD5N012,83WD3N009,83WD3N005,83CA1N018,83CA1N008,83CA1N015,83CA1N014,83FU3N013,83FU3N015,83FU3N011,83FU5N024,83MR2N051,83MR2N060,83MR2N052,83MR2N041,83IN5N018,83IN4P009,83IN5N030,83IN5N035,83CY3U005,83CY3U009,83CY3U004,83CY3N002,83SE1N009,83SE1N005,83SE2N047,83SE1N010,83HO3N003,83HO3U010,83HO3N014,83HO3U006,83UN3N002,83UN5N012,83UN5N042,83UN5N023,83LW3N005,83LW3N010,83LW4N019,83LW3N003,83CW2N004,83CW2N003,83SH3N009,83SH3N008,83SH3N012,83HE3N009,83HE3N007,83HE3N008,83LN2N005,83LN2N007,83MA2P059,83MA1N012,83MA2N022,83MA1N011,83ST5N018,83CS2N002,83GA3N006,83GA3N012,83LE3N009,83LA3U014,83LA3U012,83YE1N013,83YE2N041,83YE5N056,83YE4N055,83CN3N009,83CN1N003,83CN2N005,83CN5N020,83LO1N013,83LO2N040,83LO1N024,83LO1N021,83SV3N006,83SV3N005,83BA3N017,83BA4N019,83BA3N011,83BA2N007,83FA3N015,83FA5N025,83FA3N020,83FA2N002,83BE2N020,83BO1N023,83BO2N063,83BO1N001,83BO1N016,83SB4N019,83SB2N007,83SB2N001,83CL3N010,83CL3N012,83CL3N011,83CL5N018,83PN2N001,83PU5N023,83PO3N004,83PO3N005,83LI3U002,83RA3N006,83RA2N003,83RA3N005,83VB2N022,83VB2N013,83VB5N035,83VB2N007,83OU2N001,83OU2N008,83OU3U021,83OU3N014,83PH2N006,83JO1N017,83JO2N067,83JO1N024,83JO1N015,83GE3N001,83GE3N003,83AR2N005,83IZ5N013,83IZ5N017,83IZ3N009,83NE3U008,83JE1N001,83BR3N002,83PL3N002,83DR3N007,83DR3N012,83WA3N023,83WA2N008,83SA3N004,83SA3N007,83PR2N008,83GR3N009,83GR3N008,83GR2N001,83WH2N004,83CB2N007,83CB2N003,83MI3U011,83PI3N005,83LR2N002,83CO3N001,83PE2N017,83PE3N21,83DS3N012,83SF3N009,83SF2N001,83DL3N004,83ML2N002,83CT3N007' --highlight_basemap_counties 'Benton,Washington,Scott,Crawford,Washington,Sebastian,Yell,Logan,Franklin,Madison,Carroll,Boone,Newton,Johnson,Pope,Van Buren,Searcy,Marion,Baxter,Stone,Independence,Jackson,Randolph,Bradley,Union,Ashley'"

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
