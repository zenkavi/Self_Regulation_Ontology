subid=$1
order=$2

echo "Creating battery for subject: $subid"
echo "Order: $order"
expfactory --run --folder scanner_tasks_set$order/ --battery expfactory-battery/ --experiments motor_selective_stop_signal,stroop,dot_pattern_expectancy,survey_medley,columbia_card_task_cold --subid $subid