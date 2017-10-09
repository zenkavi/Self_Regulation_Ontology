setup-docker:
	@echo 'expfactory_token:/dev/null \n\
results_directory:/results \n\
base_directory:/work/Self_Regulation_Ontology/ \n\
dataset:Complete_07-08-2017' > Self_Regulation_Settings_docker.txt
	python setup.py install --setupfile /work/Self_Regulation_Ontology/Self_Regulation_Settings_docker.txt
