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
    openssh-client \
    python3-pip \
    python3-dev \
    python3-setuptools \
    # requirements for numpy
    libopenblas-base \
    python3-numpy \
    python3-scipy \
    # requirements for keras
    python3-h5py \
    python3-yaml \
    # requirements for pydot
    python3-pydot && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    chmod +x /imaterialist && \
    pip --no-cache-dir uninstall -y tensorflow && \
    pip3 --no-cache-dir install --upgrade \
    tensorflow-gpu \
    keras \
    matplotlib \
    pandas \
    Pillow \
    scikit-learn \
    urlib3 \
    tqdm

WORKDIR /imaterialist
