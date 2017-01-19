set -e
for exp_id in stroop
do
sed "s/{EXP_ID}/$exp_id/g" get_experiment_designs.batch | sbatch -p russpold
done

