set -e
for exp_id in writing_task
do
sed "s/{EXP_ID}/$exp_id/g" calculate_discovery_DVs.batch | sbatch -p russpold
sed "s/{EXP_ID}/$exp_id/g" calculate_validation_DVs.batch | sbatch -p russpold
done

