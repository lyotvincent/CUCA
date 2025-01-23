FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

LABEL maintainer="cyyan@mail.nankai.edu.cn"
LABEL description="Base docker image for CUCA develop environment."

ARG user=appuser
ARG PYTHON_VERSION=3.9
ARG TORCH_VERSION=2.4.1
ARG TORCHVISION_VERSION=0.19.1
ARG TORCHAUDIO_VERSION=2.4.1

ENV FORCE_CUDA="1"
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=TRUE


RUN apt update &&\
    apt install -y htop git tmux vim libvips libvips-dev openslide-tools python3-openslide libsm6 libxext6 libxrender-dev libgl1-mesa-glx wget libssl-dev libopencv-dev libspdlog-dev gcc g++ ffmpeg


RUN wget -O ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh --no-check-certificate && \ 
    chmod +x ~/miniconda.sh && \ 
    bash ~/miniconda.sh -b -p /opt/conda && \ 
    rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ &&\
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ &&\
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ &&\
    conda config --set show_channel_urls yes


RUN conda clean -ya &&\
    conda install -y -n base ipykernel --update-deps --force-reinstall &&\
    conda install -y python=${PYTHON_VERSION} conda-build &&\
    conda install -y tqdm pyyaml numpy pillow ipython cython typing typing_extensions mkl mkl-include ninja &&\
    conda install -y conda-forge::openslide-python &&\
    conda clean -ya


RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple &&\
    pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} &&\
    pip install anndata 'scanpy[leiden]' tifffile[all] &&\
    pip install paramiko parse openpyxl loguru pandas &&\
    pip install opencv-python lightgbm timm pyvips scipy scikit_learn h5py peft &&\
    pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12==24.6.* dask-cudf-cu12==24.6.* cucim-cu12==24.6.* raft-dask-cu12==24.6.* &&\
    pip cache purge


RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple &&\
    pip install transformers datasets geopandas spatialdata cellvit-light &&\
    pip install future tensorboard nvitop clearml tidecv jupyter jupyterlab &&\
    pip cache purge


RUN useradd -ms /bin/bash -u 1007 ${user} && usermod -aG sudo ${user}
USER ${user}
WORKDIR /home/${user}
# set working directory


RUN git clone https://github.com/Mahmoodlab/CONCH.git &&\
    pip install git+file:///root/CONCH &&\
    rm -r CONCH

RUN pip install hestcore -i https://pypi.Python.org/simple/ &&\
    pip cache purge

# export port 6009 on the docker
EXPOSE 6009