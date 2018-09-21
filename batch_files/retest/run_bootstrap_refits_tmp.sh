while read var_id; do
  sed "s/{VAR_ID}/$var_id/g" bootstrap_hddm_refits.batch | sbatch -p russpold
done <ddm_report_vars.txt
