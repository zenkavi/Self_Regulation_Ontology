#! /usr/bin/env python
#
# Copyright (C) 2016 Russell Poldrack <poldrack@stanford.edu>
# some portions borrowed from https://github.com/mwaskom/lyman/blob/master/setup.py


descr = """Self_Regulation_Ontology: analysis code for UH2 self-regulation project"""

import os,sys,shutil
from setuptools import setup,find_packages

DISTNAME="selfregulation"
DESCRIPTION=descr
MAINTAINER='Russ Poldrack'
MAINTAINER_EMAIL='poldrack@stanford.edu'
LICENSE='MIT'
URL='http://poldracklab.org'
DOWNLOAD_URL='https://github.com/poldrack/Self_Regulation_Ontology/'
VERSION='0.1.0.dev'

if '--setupfile' in sys.argv:
    index = sys.argv.index('--setupfile')
    sys.argv.pop(index)  # Removes the '--foo'
    setupfile = sys.argv.pop(index)  # Returns the element after the '--foo'
else:
    setupfile='Self_Regulation_Settings.txt'

if not os.path.exists(setupfile):
    print('you must first create a Self_Regulation_Settings.txt file')
    print('see Self_Regulation_Settings_example.txt for an example')
    sys.exit()

print("copying Self_Regulation_Settings.txt into distribution")

if not os.path.exists('selfregulation/data'):
    os.mkdir('selfregulation/data')
    open('selfregulation/data/__init__.py', 'w').close()

shutil.copy(setupfile,'selfregulation/data/Self_Regulation_Settings.txt')

setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    version=VERSION,
    url=URL,
    download_url=DOWNLOAD_URL,
    packages=find_packages(),
    package_data={'selfregulation': ['data/*']},
    #scripts=['bin/stowe-towels.py','bin/wash-towels.py'],
    long_description=open('README.md').read(),
    install_requires=[
        "IPython >= 5.1.0",
        "numpy >= 1.11.2",
        "scipy >= 0.18.1",
        "matplotlib >= 1.5.3",
        "scikit-learn >= 0.19.0",
        "statsmodels >= 0.6.1",
        "networkx >= 1.9.1",
	"imblearn","ipdb","fancyimpute",
        "pandas >= 0.19.1"],
    classifiers=['Intended Audience :: Science/Research',
                 'Programming Language :: Python :: 3.4',
                 'License :: OSI Approved :: BSD License',
                 'Operating System :: POSIX',
                 'Operating System :: Unix',
                 'Operating System :: MacOS'],
)
