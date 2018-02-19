while read var_id; do
  sed "s/{VAR_ID}/$var_id/g" bootstrap_hddm_flat.batch | sbatch -p russpold
done <hddm_flat_vars.txt
