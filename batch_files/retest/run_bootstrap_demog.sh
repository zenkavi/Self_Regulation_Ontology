while read var_id; do
  sed "s/{VAR_ID}/$var_id/g" bootstrap_demog.batch | sbatch -p russpold
done <demog_vars.txt
