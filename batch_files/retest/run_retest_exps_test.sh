set -e
for exp_id in eating_survey five_facet_mindfulness_survey
do
sed "s/{EXP_ID}/$exp_id/g" calculate_retest_DVs.batch | sbatch -p russpold
done

