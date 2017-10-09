# Dockerfile for behavioral prediction analysis

FROM python:3.6.3
MAINTAINER Russ Poldrack <poldrack@gmail.com>

# Update aptitude with new repo
RUN apt-get update && apt-get install -y git

# Make ssh dir
RUN mkdir /root/.ssh/

# Copy over private key, and set permissions
ADD id_rsa_docker /root/.ssh/id_rsa

# Create known_hosts
RUN touch /root/.ssh/known_hosts
# Add bitbuckets key
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

WORKDIR /workdir

RUN apt-get update && apt-get install -y default-jre gfortran

RUN wget https://cran.r-project.org/src/base/R-3/R-3.4.2.tar.gz
RUN tar zxf R-3.4.2.tar.gz
RUN cd R-3.4.2 && ./configure --enable-R-shlib=yes && make && make install


RUN pip install \
  numpy==1.13.3 \
  pandas==0.20.3 \
  scikit-learn==0.19.0 \
  imbalanced-learn==0.3.0 \
  nilearn==0.3.0 \
  scipy==0.19.1 \
  matplotlib==2.1.0 \
  networkx==2.0 \
  statsmodels==0.8.0


RUN echo 'install.packages(c("mpath","glmnet","foreach","iterators","pscl","numDeriv","doParallel"), repos="http://cran.us.r-project.org", dependencies=TRUE)' > /tmp/packages.R \
      && Rscript /tmp/packages.R
RUN rm -rf /workdir/R-3.4.2*

ENV C_INCLUDE_PATH /usr/local/lib/R/include
ENV LD_LIBRARY_PATH /usr/local/lib/R/lib

RUN pip install rpy2

RUN pip install jinja2 \
  ipdb \
  fancyimpute==0.2.0 \
  IPython==6.2.1


# Clone the conf files into the docker container
RUN git clone git@github.com:poldrack/Self_Regulation_Ontology.git
RUN mkdir /results
WORKDIR /workdir/Self_Regulation_Ontology

RUN make setup-docker
