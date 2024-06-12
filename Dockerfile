# get the development image from nvidia cuda 12.1
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel

# create workspace folder and set it as working directory
WORKDIR /usr/app/src

# Enable 32bit Packages
RUN dpkg --add-architecture i386

# Set the timezone
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/America/Chicago /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3.10 \
    git \
    gcc g++ \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libglu1 libglu1-mesa:i386 libxcursor-dev \
    libxft2 libxft2:i386 libxinerama-dev \
    openssh-client \
    xvfb xserver-xephyr \
    less

   # Install dependencies for building custom C++ ops
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libboost-python-dev \
    ninja-build

# update package lists and install git, wget, vim, libegl1-mesa-dev, and libglib2.0-0
RUN apt-get update && \
    apt-get install -y build-essential git wget vim libegl1-mesa-dev libglib2.0-0 unzip

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
RUN conda install cuda -c nvidia/label/cuda-11.8.0 -y

RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
RUN export VLLM_VERSION=0.4.0
RUN export PYTHON_VERSION=310
RUN pip install https://github.com/vllm-project/vllm/releases/download/v0.2.2/vllm-0.2.2+cu118-cp310-cp310-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118RUN pip install xformers==0.0.22.post7
RUN pip install triton

# change the working directory to the repository
WORKDIR /usr/app/src/InstantMesh
COPY . /usr/app/src/InstantMesh
COPY InstantMesh/requirements.txt ./
RUN pip install -r requirements.txt

WORKDIR /usr/app/src/DeepCAD
COPY . /usr/app/src/DeepCAD
COPY DeepCAD/requirements.txt ./
RUN pip install -r requirements.txt

WORKDIR /usr/app/src
copy app.py ./

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Run the command when the container starts
# CMD ["python", "app.py"]
# ENTRYPOINT [ "python", "app.py", "--server_name", "0.0.0.0", "--server_port", "7860" ]