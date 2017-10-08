# Dockerfile for behavioral prediction analysis

FROM python:3.6.3
MAINTAINER Russ Poldrack <poldrack@gmail.com>

# Update aptitude with new repo
RUN apt-get update && apt-get install -y git

# Make ssh dir
RUN mkdir /root/.ssh/

# Copy over private key, and set permissions
ADD id_rsa /root/.ssh/id_rsa

# Create known_hosts
RUN touch /root/.ssh/known_hosts
# Add bitbuckets key
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts


RUN apt-get update && apt-get install -y \
  r-base=3.1.1-1+deb8u1 \
  r-base-dev=3.1.1-1+deb8u1
RUN pip install \
  numpy==1.13.3 \
  pandas==0.20.3 \
  scikit-learn==0.19.0 \
  imbalanced-learn==0.3.0 \
  nilearn==0.3.0


RUN echo 'install.packages(c("mpath","glmnet","foreach","iterators","pscl","numDeriv","doParallel"), repos="http://cran.us.r-project.org", dependencies=TRUE)' > /tmp/packages.R \
      && Rscript /tmp/packages.R

WORKDIR /work

# Clone the conf files into the docker container
#RUN git clone git@github.com:poldrack/Self_Regulation_Ontology.git

#RUN echo $'expfactory_token:/dev/null \n\
#results_directory:/work/Self_Regulation_Ontology/results \n\
#base_directory:/work/Self_Regulation_Ontology/ \n\
#dataset:Complete_07-08-2017' > /work/Self_Regulation_Ontology/Self_Regulation_Settings_docker.txt

#WORKDIR /work/Self_Regulation_Ontology

#RUN python setup.py install --setupfile /work/Self_Regulation_Ontology/Self_Regulation_Settings_docker.txt
