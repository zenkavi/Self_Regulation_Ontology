subid=$1
tasks=$2

echo "Creating battery for subject: $subid"
echo "Tasks: $tasks"
expfactory --run --folder scanner_tasks/ --battery expfactory-battery/ --experiments $tasks --subid $subid