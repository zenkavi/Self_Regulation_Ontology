#!/bin/sh
function usage {
    echo "usage: $0 "
    echo "  -i      subid (mandatory)"
    echo "  -s      scanner set (integer: 1-4, mandatory)"
    echo "  -t      task"
    exit 1
}

while getopts hi:s:t: option
do
        case "${option}"
        in
        		h) usage;;
                i) subid=${OPTARG};;
                s) scanner_set=${OPTARG};;
                t) task=${OPTARG};;
        esac
done

echo "Creating battery for subject: $subid"
echo "Task: $task"
expfactory --run --folder scanner_tasks_set$scanner_set/ --battery expfactory-battery/ --experiments $task --subid $subid