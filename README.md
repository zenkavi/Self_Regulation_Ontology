# Self_Regulation_Ontology
![CircleCI](https://circleci.com/gh/poldrack/Self_Regulation_Ontology.svg?style=svg&circle-token=c2c503d9ef106e45769fa00ca689b3b10d882c9d)

This is the main reposistory for analyses and data for the UH2 Self Regulation Ontology project.

### Setting up the repository

In order to use the code, you first need to create a version of the settings file, using the following steps:

1. Copy the file "Self_Regulation_Settings_example.txt" to a new file called "Self_Regulation_Settings.txt"

2. Using your favorite text editor, edit the file to specify the location of the project directory on the line 
starting with "base directory".  For example, on my computer it looks like:
base_directory:/Users/poldrack/code/Self_Regulation_Ontology/

Note: If you do not create a settings file, one will be created by the setup.py file (see below) with default values

### Organization of the repository

Data: contains all of the original and derived data

data_preparation: code for preparing derived data

utils: utilities for loading/saving data and metadata

other directories are specific to particular analyses - for any analysis you wish to add, please give it a descriptive name along with your initials - e.g. "irt_analyses_RP"


### Setting up python environment

python setup.py install
rp2 needs to be installed
The package below needs to be installed:
git+https://github.com/IanEisenberg/dynamicTreeCut#eb822ebb32482a81519e32e944fd631fb9176b67 

The package below needs to be installed if you want to download/process raw data
git+https://github.com/IanEisenberg/expfactory-analysis
### R setup
install:

missForest
psych
lme4
qgraph

### Docker usage

to build run:
`docker build --rm -t sro .`

Mount the Data and Results directory from the host into the container at /SRO/Data and /SRO/Results respectively

To start bash in the docker container with the appropriate mounts run:
`docker run --entrypoint /bin/bash -v /home/ian/Experiments/expfactory/Self_Regulation_Ontology/Data:/SRO/Data -it sro`
