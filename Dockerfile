# This Dockerfile builds the image of the deep learning experiment
FROM cschranz/gpu-jupyter:v1.8_cuda-12.5_ubuntu-22.04
LABEL authors="Your Name <e-mail@example.com>"

# #############################################################
# ################### Custom installations ####################
# #############################################################

# apt installs
USER root
RUN apt-get update && \
    apt-get -y install apt-utils

# Copy requirements.txt and install in pip
ADD requirements.txt .
RUN pip install -r requirements.txt

# #############################################################
# ##################### Copy the content ######################
# #############################################################

# Copy the content of this repository into the directory
RUN mkdir /home/jovyan/work/reproducible_project
ADD . /home/jovyan/work/reproducible_project

# #############################################################
# ###################### Set environment ######################
# #############################################################

# fix permissions to avoid files or folders hidden in the container
USER root
RUN chown -R ${NB_USER}.${NB_GID} /home/jovyan/work/

# Switch back to user Jovyan to avoid accidental container runs as root
USER ${NB_UID}
WORKDIR ${HOME}
