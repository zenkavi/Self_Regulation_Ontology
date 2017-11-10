set -e
for exp_id in dot_pattern_expectancy two_stage_decision
do
sed "s/{EXP_ID}/$exp_id/g" calculate_retest_DVs.batch | sbatch -p russpold
done

