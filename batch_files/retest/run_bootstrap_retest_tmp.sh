while read var_id; do
  sed "s/{VAR_ID}/$var_id/g" bootstrap_retest.batch | sbatch -p russpold
done <ddm_report_vars.txt
