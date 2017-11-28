set -e
for exp_id in alcohol_drugs_survey bickel_titrator
do
sed "s/{EXP_ID}/$exp_id/g" calculate_retest_DVs.batch | sbatch -p russpold
done

