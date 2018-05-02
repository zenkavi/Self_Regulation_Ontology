set -e
for exp_id in directed_forgetting local_global_letter motor_selective_stop_signal shape_matching stim_selective_stop_signal stop_signal threebytwo 
do
sed -e "s/{EXP_ID}/$exp_id/g" -e "s/{SUBSET}/mturk_retest/g" calculate_hddm_refits.batch | sbatch
done
