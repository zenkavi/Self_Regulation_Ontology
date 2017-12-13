set -e
for exp_id in adaptive_n_back choice_reaction_time motor_selective_stop_signal shift_task simon stim_selective_stop_signal stop_signal stroop two_stage_decision 
do
sed -e "s/{EXP_ID}/$exp_id/g" -e "s/{SUBSET}/mturk_complete/g" calculate_exp_DVs.batch | sbatch --time=48:00:00 -p russpold
done
