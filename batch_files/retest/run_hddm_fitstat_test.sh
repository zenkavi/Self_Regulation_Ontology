set -e
for model in simon_base.model
do
sed "s/{MODEL}/$model/g" calculate_hddm_fitstat_complete.batch | sbatch -p russpold
done
