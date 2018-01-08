set -e
for exp_id in threebytwo # adaptive_n_back stop_signal threebytwo
do
    sed -e "s/{EXP_ID}/$exp_id/g" -e "s/{SUBSET}/mturk_complete/g" calculate_exp_DVs.batch | sbatch --time=4:00:00 -p russpold
 
    #sed -e "s/{EXP_ID}/$exp_id/g" -e "s/{SUBSET}/mturk_complete_subset1/g" calculate_exp_DVs.batch | sbatch --time=4:00:00 -p russpold
    #sed -e "s/{EXP_ID}/$exp_id/g" -e "s/{SUBSET}/mturk_complete_subset2/g" calculate_exp_DVs.batch | sbatch --time=4:00:00 -p russpold
done

