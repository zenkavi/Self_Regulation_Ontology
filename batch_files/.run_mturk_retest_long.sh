set -e
for exp_id in shift_task two_stage_decision 
do
sed -e "s/{EXP_ID}/$exp_id/g" -e "s/{SUBSET}/mturk_retest/g" $HOME/Self_Regulation_Ontology/batch_files/calculate_exp_DVs.batch | sbatch --time=48:00:00 -p russpold
done
