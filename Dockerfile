# docker build -t ziyuli/model-inference:1.0.0 .

FROM tensorflow/tensorflow:latest-gpu
# FROM python:3
# WORKDIR /usr/src/app



COPY requirements.txt ./

RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt update
RUN apt-get update
RUN apt-get install git

# RUN apt-get install -y python3.8 python3-pip

# RUN bash Miniconda3-latest-Linux-x86_64.sh
# RUN source ~/.bashrc
# RUN conda -V
# RUN conda install -c conda-forge cudatoolkit=11.1 cudnn=8.1.0
# RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# # # Add 3.7 to the available alternatives
# # RUN alternatives --install /usr/bin/python python /usr/bin/python3.8 1
# # # Set python3.7 as the default python
# # RUN alternatives --set python /usr/bin/python3.8

RUN python3 -m pip install pip

RUN python3 -V && pip --version

RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir -r requirements.txt
RUN python3 -m pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# You shall create your own requirements.txt with all the python packages you want to install.
# to build the docker, run it under the path with Dockerfile
# docker build -t [docker_image_name] .

# to run the docker, you may use the following command
# docker rm [container_name]
# docker run -it --name [container_name] --gpus "device=1" --shm-size=8gb -v /home/ziyuli:/usr/src/app [docker_image_name]
