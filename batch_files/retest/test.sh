for EXP_ID in choice_reaction_time recent_probes
do
	for SUBSET in _retest _t1
    do
    cd /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_retest_output/hddm_flat/subject_fits
    find . -type f -name "${EXP_ID}${SUBSET}_s*hddm_flat.csv" -exec cat > "../${EXP_ID}${SUBSET}_hddm_flat.csv" {} \;
    awk '!a[$0]++' "../${EXP_ID}${SUBSET}_hddm_flat.csv" > "../${EXP_ID}${SUBSET}_hddm_flat_clean.csv"
    rm ../${EXP_ID}${SUBSET}_hddm_flat.csv
    mv ../${EXP_ID}${SUBSET}_hddm_flat_clean.csv ../${EXP_ID}${SUBSET}_hddm_flat.csv
    done
done