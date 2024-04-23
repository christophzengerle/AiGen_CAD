# Using PyTorch image as the base
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel

WORKDIR /usr/app/src


ENV BUILD_WITH_CUDA True
ENV CUDA_HOME /usr/local/cuda-11.8/

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3.12.2-dev \
    git \
    gcc g++ \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgl1-mesa-glx \
    libsndfile1


# This step is necessary to use GROUNDINGDINO!

# Install dependencies for building custom C++ ops
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libboost-python-dev \
    ninja-build


WORKDIR /usr/app/src

COPY . ./DeepCAD

WORKDIR /usr/app/src/DeepCAD

RUN pip install -r requirements.txt

CMD["python", "main.py"]
