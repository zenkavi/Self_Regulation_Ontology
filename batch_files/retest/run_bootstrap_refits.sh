while read var_id; do
  sed "s/{VAR_ID}/$var_id/g" bootstrap_retest.batch | sbatch -p russpold
done <hddm_retest_vars.txt
