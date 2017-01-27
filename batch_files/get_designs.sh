set -e
#for exp_id in attention_network_task dot_pattern_expectancy motor_selective_stop_signal stop_signal stroop twobytwo ward_and_allport
for exp_id in attention_network_task dot_pattern_expectancy motor_selective_stop_signal
do
for index in 2 3 4
do
sed -e "s/{EXP_ID}/$exp_id/g" -e "s/{INDEX}/$index/g" get_experiment_designs.batch | sbatch -p russpold
done
done
