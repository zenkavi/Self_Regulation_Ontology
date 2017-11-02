while read VAR_ID
do
    sed "s/{VAR_ID}/$var_id/g" bootstrap_retest.batch | sbatch -p russpold
done < variables_exhaustive_names.txt