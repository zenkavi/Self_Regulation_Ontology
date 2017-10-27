mkdir ../../Data/Retest_09-27-2017/batch_output/hddm_model
mkdir ../../Data/Retest_09-27-2017/batch_output/hddm_data
mv *.model hddm_models
mv *.db hddm_models
mv *pcc.csv hddm_models
mv *_data.csv hddm_data
python concatenate_retest_DVs.py
