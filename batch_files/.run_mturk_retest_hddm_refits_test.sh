set -e
for exp_id in attention_network_task dot_pattern_expectancy
do
sed -e "s/{EXP_ID}/$exp_id/g" -e "s/{SUBSET}/mturk_retest/g" calculate_hddm_refits.batch | sbatch 
done
