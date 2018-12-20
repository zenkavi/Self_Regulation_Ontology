set -e
for exp_id in kirby
do
  for proptrials in 0.25 0.5 0.75
  do
    for rand in yes no
    do
      for subset in retest complete
      do
        sed -e "s/{EXP_ID}/$exp_id/g" -e "s/{SUBSET}/$subset/g" -e "s/{PROPTRIALS}/$proptrials/g" -e "s/{RAND}/$rand/g" calculate_exp_DVs_proptrials.batch | sbatch --time=02:00:00 --cpus-per-task=4
      done
    done
  done
done
