set -e
for exp_id in digit_span 
do
sed "s/{EXP_ID}/$exp_id/g" calculate_complete_DVs.batch | sbatch -p russpold
done
