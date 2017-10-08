docker-build:
	docker build -t poldrack:selfregulation .

docker-devel:
	docker run  -it -v /Users/poldrack/code/Self_Regulation_Ontology:/work/Self_Regulation_Ontology poldrack:selfregulation bash

keygen:
	ssh-keygen -f id_rsa_docker
	@echo "now add id_rsa_docker.pub to your github keys"
