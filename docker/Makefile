docker2singularity:
	docker run -v /var/run/docker.sock:/var/run/docker.sock -v /home/poldrack/singularity:/output --privileged -t --rm singularityware/docker2singularity selfregulation

docker-build:
	docker build -t selfregulation .

docker-devel:
	docker run  -it -v /Users/poldrack/code/Self_Regulation_Ontology:/work/Self_Regulation_Ontology selfregulation bash

keygen:
	ssh-keygen -f id_rsa_docker
	@echo "now add id_rsa_docker.pub to your github keys"
