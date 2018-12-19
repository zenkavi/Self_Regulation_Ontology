set -e
for exp_id in shift_task two_stage_decision
do
  for proptrials in 0.25 0.5 0.75
  do
    for rand in True False
    do
      sed -e "s/{EXP_ID}/$exp_id/g" -e "s/{SUBSET}/mturk_retest/g" -e "s/{PROPTRIALS}/$proptrials/g" -e "s/{RAND}/$rand/g" calculate_exp_DVs_proptrials.batch | sbatch --time=48:00:00 --cpus-per-task=8
    done
  done
done
