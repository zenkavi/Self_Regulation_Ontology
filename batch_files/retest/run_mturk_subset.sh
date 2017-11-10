set -e
for exp_id in dot_pattern_expectancy two_stage_decision 
do
sed "s/{EXP_ID}/$exp_id/g" calculate_complete_DVs.batch | sbatch -p russpold
done
