set -e
for exp_id in adaptive_n_back
do
sed "s/{EXP_ID}/$exp_id/g" calculate_hddm_flat.batch | sbatch -p russpold
done
