FROM nvcr.io/nvidia/tensorflow:17.10

VOLUME ["/imaterialist"]

# Run the copied file and install some dependencies
RUN apt update -qq && \
    apt install --no-install-recommends -y \
    # install essentials
    build-essential \
    g++ \
    git \
    graphviz \
    cython \
    openssh-client \
    python3-pip \
    python3-dev \
    python3-setuptools \
    # requirements for numpy
    libopenblas-base \
    # requirements for keras
    python3-yaml \
    # requirements for pydot
    python3-pydot && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    chmod +x /imaterialist && \
    pip --no-cache-dir uninstall -y tensorflow && \
    python3 -m pip --no-cache-dir install --upgrade pip && \
    python3 -m pip --no-cache-dir install --upgrade \
    cython \
    h5py \
    scipy \
    tensorflow-gpu \
    keras \
    matplotlib \
    numpy \
    pandas \
    Pillow \
    scikit-learn \
    scikit-image \
    urllib3 \
    tqdm

WORKDIR /imaterialist
ENV CUDA_VISIBLE_DEVICES 0
