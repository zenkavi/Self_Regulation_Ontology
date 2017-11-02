while read VAR_ID
do
    sed "s/{VAR_ID}/$var_id/g" bootstrap_retest.batch | sbatch -p russpold
done < head -n 1 /oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Retest_09-27-2017/variables_exhaustive.csv | sed s/,/'\n'/g
