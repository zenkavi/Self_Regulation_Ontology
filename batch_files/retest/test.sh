set -e
for exp_id in choice_reaction_time
do
sed -e "s/{EXP_ID}/$exp_id/g" -e "s/{DATA_PATH}/\/oak\/stanford\/groups\/russpold\/users\/zenkavi\/Self_Regulation_Ontology\/Data\/Retest_01\-23\-2018\/t1_data\/Individual_Measures\//g" -e "s/{SUBSET}/t1/g" calculate_hddm_flat.batch | sbatch -p russpold
done
