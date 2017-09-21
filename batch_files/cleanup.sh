mv *.model /scratch/users/ieisenbe/Self_Regulation_Ontology/Data/hddm_models
mv *.db /scratch/users/ieisenbe/Self_Regulation_Ontology/Data/hddm_models
mv *pcc.csv /scratch/users/ieisenbe/Self_Regulation_Ontology/Data/hddm_models
mv *_data.csv /scratch/users/ieisenbe/Self_Regulation_Ontology/Data/hddm_data
python concatenate_DVs.py
mv output/mturk*.json $SCRATCH/Self_Regulation_Ontology/Data
