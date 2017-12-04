#!/bin/bash
docker build -t nipype_image Docker
rm -f singularity_images/*img
docker run -v /var/run/docker.sock:/var/run/docker.sock -v /home/ian/Experiments/expfactory/Self_Regulation_Ontology/singularity_images:/output --privileged -t --rm singularityware/docker2singularity sro
echo Finished Conversion
cd singularity_images
bash transfer_image.sh
echo Finished Transfer
