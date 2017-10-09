define setup
expfactory_token:/dev/null
results_directory:/results
base_directory:/workdir/Self_Regulation_Ontology
dataset:Complete_07-08-2017
endef
export setup

setup-docker:
	@echo  "$$setup">Self_Regulation_Settings.txt
	#python setup.py install 
