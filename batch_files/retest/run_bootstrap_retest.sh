while read var_id; do
  sed "s/{VAR_ID}/$var_id/g" bootstrap_retest.batch | sbatch -p russpold
done <retest_report_vars_test.txt
