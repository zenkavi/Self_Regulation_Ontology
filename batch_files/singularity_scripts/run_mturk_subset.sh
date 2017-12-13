set -e
for exp_id in simon
do
sed -e "s/{EXP_ID}/$exp_id/g" -e "s/{SUBSET}/mturk_complete/g" calculate_exp_DVs.batch | sbatch --time=20:00:00 -p russpold
done
