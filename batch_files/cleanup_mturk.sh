mv *.model /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/hddm_models
mv *.db /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/hddm_models
mv *pcc.csv /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/hddm_models
mv *_data.csv /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/hddm_data
python concatenate_mturk_DVs.py
mv output/mturk*.json /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/
