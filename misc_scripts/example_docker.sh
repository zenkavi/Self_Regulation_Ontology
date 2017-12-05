docker run --rm  -ti sro_dataprep


# calculate exp dvs
data_loc=/home/ian/tmp/
output=/media/Data/Ian/Experiments/expfactory/Self_Regulation_Ontology/batch_files/container_scripts/output
docker run --rm  \
--mount type=bind,src=$data_loc,dst=/Data \
--mount type=bind,src=$output,dst=/output \
-ti sro batch_files/calculate_exp_DVs.py stim_selective_stop_signal mturk_complete --out_dir /output 
