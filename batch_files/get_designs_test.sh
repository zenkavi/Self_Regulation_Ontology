set -e
for exp_id in attention_network_task
do
for index in 3 4
do
sed -e "s/{EXP_ID}/$exp_id/g" -e "s/{INDEX}/$index/g" get_experiment_designs.batch | sbatch -p russpold
done
done
