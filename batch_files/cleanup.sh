mv *.model hddm_models
mv *.db hddm_models
mv *pcc.csv hddm_models
mv *_data.csv hddm_data
python concatenate_DVs.py
mv output/mturk*.json $SCRATCH/Self_Regulation_Ontology/Data