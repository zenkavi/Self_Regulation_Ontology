source activate $HOME/conda_envs/SRO
python ../helper_funcs/concatenate_mturk_DVs.py
mv output/*.model /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_hddm/hddm_models
mv output/*.db /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_hddm/hddm_models
mv output/*pcc.csv /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_hddm/hddm_models
mv output/mturk*.json /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/
