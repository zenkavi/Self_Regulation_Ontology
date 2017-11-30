set -e
for exp_id in adaptive_n_back attention_network_task choice_reaction_time directed_forgetting dot_pattern_expectancy local_global_letter motor_selective_stop_signal recent_probes shape_matching  simon stim_selective_stop_signal stop_signal stroop threebytwo 
do
sed "s/{EXP_ID}/$exp_id/g" calculate_complete_DVs.batch | sbatch -p russpold
#sed "s/{EXP_ID}/$exp_id/g" calculate_validation_DVs.batch | sbatch -p russpold
#sed "s/{EXP_ID}/$exp_id/g" calculate_discovery_DVs.batch | sbatch -p russpold
done
