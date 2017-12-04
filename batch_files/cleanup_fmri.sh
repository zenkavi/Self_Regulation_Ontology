python helper_funcs/concatenate_fmri_DVs.py
mv *.model /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/Data/fmri_hddm/hddm_models
mv *.db /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/Data/fmri_hddm/hddm_models
mv *pcc.csv /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/Data/fmri_hddm/hddm_models
mv *_data.csv /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/Data/fmri_hddm/hddm_data
python concatenate_fmri_DVs.py
mv output/fmri*.json /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/Data/
