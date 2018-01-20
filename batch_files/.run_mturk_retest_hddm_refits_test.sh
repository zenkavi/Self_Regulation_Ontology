set -e
for exp_id in adaptive_n_back directed_forgetting choice_reaction_time local_global_letter
do
sed -e "s/{EXP_ID}/$exp_id/g" -e "s/{SUBSET}/mturk_retest/g" calculate_hddm_refits.batch | sbatch 
done
