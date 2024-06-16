# get the development image from nvidia cuda 12.1
FROM nvidia/cuda:12.1.0-devel-ubuntu20.04
# FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
# FROM nvidia/cuda:11.8.0-devel-ubuntu20.04
# FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

WORKDIR /usr/app/src

# Set the timezone
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa 
RUN apt-get update && \
    apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/America/Chicago /etc/localtime && \ 
    dpkg-reconfigure --frontend noninteractive tzdata 

# update package lists and install git, wget, vim, libegl1-mesa-dev, and libglib2.0-0
RUN apt-get update && \
    apt-get install -y build-essential git wget vim libegl1-mesa-dev libglib2.0-0 unzip ninja-build


# install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /usr/app/src/miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh

# update PATH environment variable
ENV PATH="/usr/app/src/miniconda3/bin:${PATH}"

# initialize conda
RUN conda init bash

# create and activate conda environment
RUN conda create -n instantmesh python=3.10 && echo "source activate instantmesh" > ~/.bashrc
ENV PATH /usr/app/src/miniconda3/envs/instantmesh/bin:$PATH

RUN conda install Ninja
# RUN conda install cuda -c nvidia/label/cuda-11.8.0 -y
RUN conda install cuda -c nvidia/label/cuda-12.1.0 -y

ENV BUILD_WITH_CUDA=True
ENV CUDA=12.1
ENV PATH=/usr/local/cuda-$CUDA/bin${PATH:+:${PATH}}
ENV CUDA_PATH=/usr/local/cuda-$CUDA
ENV CUDA_HOME=/usr/local/cuda-$CUDA
ENV LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-$CUDA/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV NVCC=/usr/local/cuda-$CUDA/bin/nvcc
ENV CFLAGS="-I$CUDA_HOME/include $CFLAGS"


RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu121
RUN pip install triton


# RUN pip install https://github.com/vllm-project/vllm/releases/download/v0.4.0/vllm-0.4.0+cu118-cp310-cp310-manylinux1_x86_64.whl

# change the working directory to the repository
WORKDIR /usr/app/src/InstantMesh

# other dependencies
COPY ./requirements.txt ./
RUN pip install -r requirements.txt
RUN rm -rf requirements.txt

# COPY . /usr/app/src/instantmesh

# EXPOSE 7860
# ENV GRADIO_SERVER_NAME=0.0.0.0

ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024

# Run the command when the container starts
# CMD ["python", "app.py"]

CMD ["python", "app_access_point.py"]