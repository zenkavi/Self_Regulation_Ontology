set -e
for exp_id in adaptive_n_back 
do
sed -e "s/{EXP_ID}/$exp_id/g" -e "s/{DATA_PATH}/\/oak\/stanford\/groups\/russpold\/users\/zenkavi\/Self_Regulation_Ontology\/Data\/Retest_01\-23\-2018\/Individual_Measures\//g" -e "s/{SUBSET}/retest/g" calculate_hddm_flat.batch | sbatch -p russpold
done
