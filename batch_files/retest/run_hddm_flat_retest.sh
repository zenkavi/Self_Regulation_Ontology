set -e
for exp_id in adaptive_n_back attention_network_task choice_reaction_time directed_forgetting dot_pattern_expectancy local_global_letter motor_selective_stop_signal recent_probes shape_matching simon stim_selective_stop_signal stop_signal stroop threebytwo
do
sed "s/{EXP_ID}/$exp_id/g" -e "s/{DATA_PATH}/[/oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Retest_01-23-2018/Individual_Measures]/g" -e "s/{SUBSET}/retest/g" calculate_hddm_flat.batch | sbatch -p russpold
done
