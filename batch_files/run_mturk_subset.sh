set -e
for exp_id in adaptive_n_back stop_signal threebytwo
do
    sed -e "s/{EXP_ID}/$exp_id/g" -e "s/{SUBSET}/mturk_complete/g" $HOME/Self_Regulation_Ontology/batch_files/calculate_exp_DVs.batch | sbatch --time=168:00:00 -p russpold
 
    sed -e "s/{EXP_ID}/$exp_id/g" -e "s/{SUBSET}/mturk_complete_subset1/g" $HOME/Self_Regulation_Ontology/batch_files/calculate_exp_DVs.batch | sbatch --time=168:00:00 -p russpold
    sed -e "s/{EXP_ID}/$exp_id/g" -e "s/{SUBSET}/mturk_complete_subset2/g" $HOME/Self_Regulation_Ontology/batch_files/calculate_exp_DVs.batch | sbatch --time=168:00:00 -p russpold
done

