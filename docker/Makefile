docker-clean:
	-sudo rm ~/singularity/*
	-docker stop $(docker ps -a -q)
	-docker rm $(docker ps -a -q)
	-docker images -q --filter dangling=true | xargs docker rmi

docker2singularity:
	docker run -v /var/run/docker.sock:/var/run/docker.sock -v /home/poldrack/singularity:/output --privileged -t --rm singularityware/docker2singularity selfregulation

# this adds a character to the RUN command
# # which causes all commands below to be re-excuted
docker-build:
	cat Dockerfile | sed 's/RUN echo "x/RUN echo "xx/'>Dockerfile.tmp
	cp Dockerfile.tmp Dockerfile
	docker build -t selfregulation .

docker-devel:
	docker run  -it -v /Users/poldrack/code/Self_Regulation_Ontology:/work/Self_Regulation_Ontology selfregulation bash

keygen:
	ssh-keygen -f id_rsa_docker
	@echo "now add id_rsa_docker.pub to your github keys"

copy:
	scp ~/singularity/selfregulation-*.img wrangler.tacc.utexas.edu:/work/01329/poldrack/stampede2/singularity_images

shell:
	docker run -v /tmp/results:/workdir/Self_Regulation_Ontology/results -it selfregulation /bin/bash

jupyter:
	docker run -ti -v /tmp/results:/workdir/Self_Regulation_Ontology/results -p 7777:8888 selfregulation jupyter notebook /workdir/Self_Regulation_Ontology/prediction_analyses/PredictionResults.ipynb
