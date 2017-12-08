set -e
for exp_id in attention_network_task shape_matching threebytwo 
do
sed -e "s/{EXP_ID}/$exp_id/g" -e "s/{SUBSET}/mturk_complete/g" calculate_exp_DVs.batch | sbatch --qos=long --time=80:00:00
done
