subid=$1
order=$2

echo "Creating battery for subject: $subid"
echo "Order: $order"
expfactory --run --folder scanner_tasks_set$order/ --battery expfactory-battery/ --experiments stop_signal,attention_network_task,twobytwo,columbia_card_task_hot --subid $subid