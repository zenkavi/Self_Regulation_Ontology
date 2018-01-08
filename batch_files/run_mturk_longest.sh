set -e
for exp_id in attention_network_task directed_forgetting dot_pattern_expectancy local_global_letter shape_matching recent_probes threebytwo 
do
sed -e "s/{EXP_ID}/$exp_id/g" -e "s/{SUBSET}/mturk_complete/g" calculate_exp_DVs.batch | sbatch --time=80:00:00 -p russpold
done
