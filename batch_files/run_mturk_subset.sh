set -e
for exp_id in stroop 
do
sed -e "s/{EXP_ID}/$exp_id/g" -e "s/{SUIBSET}/mturk_complete/g" calculate_exp_DVs.batch | sbatch -p russpold
done
