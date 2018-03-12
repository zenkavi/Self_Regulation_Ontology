set -e
#for exp_id in directed_forgetting local_global_letter shape_matching stop_signal threebytwo 
for exp_id in motor_selective_stop_signal
do
    sed -e "s/{EXP_ID}/$exp_id/g" -e "s/{SUBSET}/mturk_complete/g" calculate_exp_DVs.batch | sbatch --time=120:00:00
    sed -e "s/{EXP_ID}/$exp_id/g" -e "s/{SUBSET}/mturk_retest/g" calculate_exp_DVs.batch | sbatch --time=120:00:00
done

