cd /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_retest_output/bootstrap_output
cat *.csv > /oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Retest_02-03-2018/Local/bootstrap_merged.csv
cd /oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Retest_02-03-2018/Local
awk '!a[$0]++' bootstrap_merged.csv > bootstrap_merged_clean.csv
rm bootstrap_merged.csv
mv bootstrap_merged_clean.csv ./bootstrap_merged.csv
gzip bootstrap_merged.csv
