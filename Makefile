setup-docker:
	@echo 'expfactory_token:/dev/null \n
results_directory:/results \n
base_directory:/workdir/Self_Regulation_Ontology/ \n
dataset:Complete_07-08-2017' > Self_Regulation_Settings.txt
	python setup.py install 
