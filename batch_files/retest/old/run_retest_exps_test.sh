set -e
for exp_id in shift_task
do
sed "s/{EXP_ID}/$exp_id/g" calculate_retest_DVs.batch | sbatch -p russpold
done

