set -e
for exp_id in threebytwo
do
sed "s/{EXP_ID}/$exp_id/g" calculate_discovery_DVs.batch | sbatch -p russpold
done

