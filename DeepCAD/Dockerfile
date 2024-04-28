# Using PyTorch image as the base
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel

WORKDIR /usr/app/src


ENV BUILD_WITH_CUDA True
ENV CUDA_HOME /usr/local/cuda/
ENV DISPLAY 192.168.178.29:0.0 


# Enable 32bit Packages
RUN dpkg --add-architecture i386

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3.10 \
    git \
    gcc g++ \
    openssh-client \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgl1-mesa-glx \
    libsndfile1 \
    libxext6:i386 \
    libxft2 libxft2:i386 lib32ncurses6 \
    libglu1 libglu1-mesa:i386


    
# Install dependencies for building custom C++ ops
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libboost-python-dev \
    ninja-build



WORKDIR /usr/app/src/DeepCAD

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN rm -rf requirements.txt

# Install PythonOCC
RUN conda install -n base conda-libmamba-solver -y
# RUN conda config --set solver libmamba
RUN conda install -c conda-forge pythonocc-core=7.7.2 --solver=libmamba -y

# Install PointNet2 ops
RUN pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
