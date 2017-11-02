while read var_id
do
    sed "s/{VAR_ID}/$var_id/g" bootstrap_retest.batch | sbatch -p russpold
done < variables_exhaustive_names_test.txt