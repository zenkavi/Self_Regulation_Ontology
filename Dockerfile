# Dockerfile for Self Regulation Ontology repo

FROM python:3.5.3
MAINTAINER Russ Poldrack <poldrack@gmail.com>

RUN apt-get update && apt-get install -y default-jre gfortran

# installing R
RUN wget https://cran.r-project.org/src/base/R-3/R-3.4.2.tar.gz
RUN tar zxf R-3.4.2.tar.gz
RUN cd R-3.4.2 && ./configure --enable-R-shlib=yes && make && make install

# installing R packages
RUN echo 'install.packages(c( \
  "doParallel", \
  "dplyr", \
  "foreach", \
  "iterators", \
  "glmnet", \
  "missForest", \
  "mpath", \
  "numDeriv", \
  "psych", \
  "pscl", \
  "tidyr" \
  ), \
  repos="http://cran.us.r-project.org", dependencies=TRUE)' > /tmp/packages.R && \
  Rscript /tmp/packages.R && \
  rm -rf /workdir/R-3.4.2*

# installing python packages
RUN pip install \
  cython==0.27.3 \ 
  git+https://github.com/IanEisenberg/dynamicTreeCut#eb822ebb32482a81519e32e944fd631fb9176b67 \
  imbalanced-learn==0.3.0 \
  ipdb \ 
  IPython==6.2.1 \
  Jinja2==2.9.6 \
  matplotlib==2.1.0 \
  networkx==2.0 \
  nilearn==0.3.0 \
  numpy==1.13.3 \
  pandas==0.20.3 \
  python-igraph==0.7.1.post6 \
  scipy==0.19.1 \
  scikit-learn==0.19.0 \
  seaborn==0.7.1 \
  statsmodels==0.8.0 \
  svgutils==0.3.0 \
  jupyter

RUN pip install hdbscan==0.8.10 
# set up rpy2
ENV C_INCLUDE_PATH /usr/local/lib/R/include
ENV LD_LIBRARY_PATH /usr/local/lib/R/lib
ENV IPYTHONDIR /tmp
# install more python packages that failed in first install
RUN pip install \
    fancyimpute==0.2.0 \
    rpy2==2.8.5

  
# Copy the directory (except Data and Results) into the docker container
ADD . /SRO
RUN mkdir /SRO/Data
RUN mkdir /SRO/Results
RUN mkdir /expfactory_token

# Create a settings file
RUN echo "expfactory_token:/expfactory_token/expfactory_token.txt" >> /SRO/selfregulation/data/Self_Regulation_Settings.txt
RUN echo "base_directory:/SRO" >> /SRO/selfregulation/data/Self_Regulation_Settings.txt
RUN echo "results_directory:/Results" >> /SRO/selfregulation/data/Self_Regulation_Settings.txt
RUN echo "data_directory:/Data" >> /SRO/selfregulation/data/Self_Regulation_Settings.txt

RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py

WORKDIR /SRO
ENTRYPOINT ["python"]