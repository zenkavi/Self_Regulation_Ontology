subid=$1
tasks=$2

echo "Creating battery for subject: $subid"
expfactory --run --folder scanner_tasks/ --battery ~/Experiments/expfactory/expfactory-battery/ --experiments $tasks --subid $subid