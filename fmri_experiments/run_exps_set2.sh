subid=$1

echo "Creating battery for subject: $subid"
expfactory --run --folder scanner_tasks/ --battery expfactory-battery/ --experiments motor_selective_stop_signal,stroop,dot_pattern_expectancy,survey_medley,columbia_card_task_cold --subid $subid