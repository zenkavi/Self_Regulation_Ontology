set -e
for ppc_data_dir in num_samples_test refit retest_flat retet_hierarchical t1_flat t1_hierarchical
do
sed -e "s/{PPC_DATA_DIR}/$ppc_data_dir/g" calculate_hddm_kl.batch | sbatch
done
