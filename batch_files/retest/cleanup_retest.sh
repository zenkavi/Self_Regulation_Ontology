mkdir ../../Data/Retest_09-27-2017/batch_output/hddm_models
mkdir ../../Data/Retest_09-27-2017/batch_output/hddm_data
mv *.model ../../Data/Retest_09-27-2017/batch_output/hddm_models
mv *.db ../../Data/Retest_09-27-2017/batch_output/hddm_models
mv *pcc.csv ../../Data/Retest_09-27-2017/batch_output/hddm_models
mv *_data.csv ../../Data/Retest_09-27-2017/batch_output/hddm_data
python concatenate_retest_DVs.py
