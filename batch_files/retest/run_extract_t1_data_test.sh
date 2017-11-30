set -e
for dataset in variables_exhaustive.csv
do
sed "s/{DATASET}/$dataset/g" extract_t1_data.batch | sbatch -p russpold
done
