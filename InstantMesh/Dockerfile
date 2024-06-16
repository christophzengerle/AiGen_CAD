# get the development image from nvidia cuda 12.1
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

ENV BUILD_WITH_CUDA=True
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH

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
    apt-get install -y build-essential git wget vim libegl1-mesa-dev libglib2.0-0 unzip



RUN conda install Ninja
RUN pip install "git+https://github.com/NVlabs/nvdiffrast/"
# RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install xformers==0.0.22.post4 --index-url https://download.pytorch.org/whl/cu118
RUN pip install triton

# change the working directory to the repository
WORKDIR /usr/app/src/InstantMesh

# other dependencies
COPY ./requirements.txt ./
RUN pip install -r requirements.txt
RUN rm -rf requirements.txt

# COPY . /usr/app/src/instantmesh

# EXPOSE 7860
ENV GRADIO_SERVER_NAME=0.0.0.0

ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024

# Run the command when the container starts
# CMD ["python", "app.py"]

CMD ["python", "app_access_point.py"]