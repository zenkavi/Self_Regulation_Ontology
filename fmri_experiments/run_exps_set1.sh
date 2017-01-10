subid=$1

echo "Creating battery for subject: $subid"
echo "Tasks: $tasks"
expfactory --run --folder scanner_tasks/ --battery expfactory-battery/ --experiments stop_signal,attention_network_task,ward_and_allport,twobytwo --subid $subid