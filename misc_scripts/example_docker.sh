docker run --rm  -ti sro_dataprep


# calculate exp dvs
data_loc=/home/ian/tmp/
output=/home/ian/tmp
docker run --rm  \
--mount type=bind,src=$data_loc,dst=/Data \
--mount type=bind,src=$output,dst=/output \
-ti sro_dataprep \
python batch_files/helper_funcs/calculate_exp_DVs.py grit_scale_survey mturk_complete --out_dir /output

# save data
data_loc=/home/ian/tmp
output=/home/ian/tmp
docker run --rm  \
--mount type=bind,src=$data_loc,dst=/Data \
--mount type=bind,src=$output,dst=/SRO/Data \
-ti sro_dataprep python data_preparation/mturk_save_data.py

# run analysis
data_loc=/home/ian/tmp
output=/home/ian/tmp
exp_id=simon
subset=mturk_complete
docker run --rm  \
--mount type=bind,src=$data_loc,dst=/Data \
--mount type=bind,src=$output,dst=/SRO/Data \
-ti sro_dataprep python /SRO/batch_files/helper_funcs/calculate_exp_DVs.py ${exp_id} ${subset} --out_dir ${output_loc} --hddm_samples 5000 --hddm_burn 2500 --thin 1