# FROM nvcr.io/nvidia/l4t-tensorflow:r32.7.1-tf2.7-py3
From nvcr.io/nvidia/l4t-tensorflow:r34.1.0-tf2.8-py3

WORKDIR /deepQLearning

# RUN apt-get -y update && \
#     apt-get -y upgrade && \
#     apt-get -y install git

RUN python3 -m pip install --upgrade pip &&\
    pip3 install gym==0.19.0 atari_py opencv_python

COPY game_roms ./game_roms

RUN python3 -m atari_py.import_roms ./game_roms
