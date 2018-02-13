#!/bin/bash
#
#SBATCH -J hddm_flat
#SBATCH --array=1-900%10

#SBATCH --time=4:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --qos=russpold
#SBATCH -p russpold

# Outputs ----------------------------------
#SBATCH -o /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_retest_output/hddm_flat/.out/%A-%a.out
#SBATCH -e /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_retest_output/hddm_flat/.err/%A-%a.err
#SBATCH --mail-user=zenkavi@stanford.edu
#SBATCH --mail-type=FAIL
# ------------------------------------------
source /home/zenkavi/.bash_profile
source activate SRO

eval $( sed "${SLURM_ARRAY_TASK_ID}q;d" xaa )
