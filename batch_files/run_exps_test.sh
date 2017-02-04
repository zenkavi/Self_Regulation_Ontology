set -e
for exp_id in time_perspective_survey
do
sed "s/{EXP_ID}/$exp_id/g" calculate_discovery_DVs.batch | sbatch -p russpold
sed "s/{EXP_ID}/$exp_id/g" calculate_validation_DVs.batch | sbatch -p russpold
done

