set -e
for task in threebytwo
do
sed -e "s/{TASK}/$task/g" -e "s/{MODEL_DIR}/\/oak\/stanford\/groups\/russpold\/users\/ieisenbe\/Self_Regulation_Ontology\/behavioral_data\/mturk_retest_output\/hddm_flat\/subject_fits\//g" -e "s/{SUB_ID_DIR}/\/oak\/stanford\/groups\/russpold\/users\/zenkavi\/Self_Regulation_Ontology\/Data\/Retest_03-29-2018\/Individual_Measures\//g" -e "s/{OUT_DIR}/\/oak\/stanford\/groups\/russpold\/users\/ieisenbe\/Self_Regulation_Ontology\/behavioral_data\/mturk_retest_output\/hddm_fitstat\//g" -e "s/{SUBSET}/retest/g" -e "s/{PARALLEL}/no/g" -e "s/{HDDM_TYPE}/flat/g" -e "s/{SAMPLES}/500/g" calculate_hddm_fitstat.batch | sbatch
done
