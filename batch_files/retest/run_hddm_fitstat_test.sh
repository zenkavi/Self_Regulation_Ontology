set -e
for model in adaptive_n_back_base.model
do
sed "s/{MODEL}/$model/g" calculate_hddm_fitstat.batch | sbatch -p russpold
done
