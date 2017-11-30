cd /oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Retest_11-27-2017/batch_output/bootstrap_output
cat *.csv > ../../Local/bootstrap_merged.csv
cd ../../Local
awk '!a[$0]++' bootstrap_merged.csv > bootstrap_merged_clean.csv
rm bootstrap_merged.csv
mv bootstrap_merged_clean.csv ./bootstrap_merged.csv
