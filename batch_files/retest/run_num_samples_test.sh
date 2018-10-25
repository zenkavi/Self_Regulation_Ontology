set -e
for task in choice_reaction_time local_global_letter
do
  for samples in 10 25 50 100 250 500
  do
    sed -e "s/{TASK}/$task/g" -e "s/{MODEL_DIR}/\/oak\/stanford\/groups\/russpold\/users\/ieisenbe\/Self_Regulation_Ontology\/behavioral_data\/mturk_retest_output\/hddm_refits\//g" -e "s/{SUB_ID_DIR}/\/oak\/stanford\/groups\/russpold\/users\/zenkavi\/Self_Regulation_Ontology\/Data\/Retest_03-29-2018\/t1_data\/Individual_Measures\//g" -e "s/{OUT_DIR}/\/oak\/stanford\/groups\/russpold\/users\/ieisenbe\/Self_Regulation_Ontology\/behavioral_data\/mturk_retest_output\/hddm_fitstat\/num_samples_tests\//g" -e "s/{SUBSET}/refit/g" -e "s/{PARALLEL}/yes/g" -e "s/{HDDM_TYPE}/hierarchical/g" -e "s/{SAMPLES}/$samples/g" calculate_hddm_fitstat.batch | sbatch
  done
done
