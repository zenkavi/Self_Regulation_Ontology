# Self_Regulation_Ontology
[![CircleCI](https://circleci.com/gh/poldrack/Self_Regulation_Ontology.svg?style=svg&circle-token=c2c503d9ef106e45769fa00ca689b3b10d882c9d)]

This is the main reposistory for analyses and data for the UH2 Self Regulation Ontology project.

### Setting up the repository

In order to use the code, you first need to create a version of the settings file, using the following steps:

1. Copy the file "Self_Regulation_Settings_example.txt" to a new file called "Self_Regulation_Settings.txt"

2. Using your favorite text editor, edit the file to specify the location of the project directory on the line 
starting with "base directory".  For example, on my computer it looks like:

base_directory:/Users/poldrack/code/Self_Regulation_Ontology/


### Organization of the repository

Data: contains all of the original and derived data

data_preparation: code for preparing derived data

utils: utilities for loading/saving data and metadata

other directories are specific to particular analyses - for any analysis you wish to add, please give it a descriptive name along with your initials - e.g. "irt_analyses_RP"


### Setting up python environment

Use the environment.yml file with anaconda: conda env create -f environment.yml

After doing that, you must install expanalysis in the same environment.
- Clone expanalysis from: https://github.com/IanEisenberg/expfactory-analysis
- Enter expanalysis and enter "pip install -e ."

Finally you must install the selfregulation python: python setup.py install

### R setup
install:

missForest
psych
lme4
qgraph
