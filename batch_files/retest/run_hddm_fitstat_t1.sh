set -e
for task in adaptive_n_back ttention_network_task choice_reaction_time directed_forgetting dot_pattern_expectancy local_global_letter recent_probes shape_matching simon stim_selective_stop_signal stop_signal stroop threebytwo
do
sed -e "s/{TASK}/$task/g" -e "s/{MODEL_DIR}/\/oak\/stanford\/groups\/russpold\/users\/ieisenbe\/Self_Regulation_Ontology\/behavioral_data\/mturk_complete_output\//g" -e "s/{SUB_ID_DIR}/\/oak\/stanford\/groups\/russpold\/users\/zenkavi\/Self_Regulation_Ontology\/Data\/Complete_03-29-2018\/Individual_Measures\//g" -e "s/{OUT_DIR}/\/oak\/stanford\/groups\/russpold\/users\/ieisenbe\/Self_Regulation_Ontology\/behavioral_data\/mturk_retest_output\/hddm_fitstat\/t1_hierarchical\//g" -e "s/{SUBSET}/t1/g" -e "s/{PARALLEL}/yes/g" -e "s/{HDDM_TYPE}/hierarchical/g" -e "s/{SAMPLES}/50/g" -e "s/{LOAD_PPC}/False/g" calculate_hddm_fitstat.batch | sbatch
done
