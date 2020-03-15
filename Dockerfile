FROM ubuntu:18.04

# Update Software repository
RUN apt-get update
# set the correct locale and encoding
RUN apt-get install -y locales locales-all
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

# Install tindetheus dependencies
RUN apt-get install -y git vim nano python3-dev python3-numpy python3-scipy python3-pip python3-venv wget perl unzip libsm6 libxrender1

# copy the cloned tindedetheus repo
COPY . tindetheus

# Installl tindetheus
# RUN git clone --recursive https://github.com/cjekel/tindetheus.git && cd tindetheus && python3 -m pip install --upgrade pip && python3 -m pip install --upgrade -r requirements.txt
RUN cd tindetheus && python3 -m pip install --upgrade pip && python3 -m pip install --upgrade -r requirements.txt

# create tinder folder
RUN mkdir tinder
