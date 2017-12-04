sshfs sherlock:/oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data /mnt/temp
rm -f /mnt/temp/singularity_images/*img
cp *img /mnt/temp/singularity_images/
sudo umount /mnt/temp