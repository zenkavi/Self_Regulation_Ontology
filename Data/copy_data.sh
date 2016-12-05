sshfs sherlock:/scratch/users/ieisenbe/ /mnt/Sherlock_Scratch
#yes | cp -rf /mnt/Sherlock_Scratch/Self_Regulation_Ontology/Data/worker_lookup.json .
#yes | cp -rf /mnt/Sherlock_Scratch/Self_Regulation_Ontology/Data/worker_counts.json .
#yes | cp -rf /mnt/Sherlock_Scratch/Self_Regulation_Ontology/Data/worker_pay.json .
yes | cp -rf /mnt/Sherlock_Scratch/Self_Regulation_Ontology/Data/mturk_discovery_DV.json Local
yes | cp -rf /mnt/Sherlock_Scratch/Self_Regulation_Ontology/Data/mturk_discovery_DV_valence.json Local
#yes | cp -rf /mnt/Sherlock_Scratch/Self_Regulation_Ontology/Data/*post* .
sudo umount /mnt/Sherlock_Scratch