# IMaterialist classifier

## Description

## Prerequisites

You will need the following things properly installed on your computer.

* [Docker](https://www.docker.com/)

## Installation

* `git clone https://github.com/smorzhov/imaterialist.git`

## Running

Remember that Docker container has the Python version 3.5.2!

1. If you are planning to use nvidia-docker, you need to build nvidia-docker image first. Otherwise, you can skip this step
    ```bash
    nvidia-docker build -t sm_keras_tf_py3:gpu .
    ```
    Run container
    ```bash
    nvidia-docker run -v $PWD/src:/imaterialist -dt --name imc sm_keras_tf_py3:gpu /bin/bash
    ```
2. Download test, train and validation data
    ```
    nvidia-docker exec imc python3 download.py data/test.json data/test
    nvidia-docker exec imc python3 download.py data/train.json data/train
    nvidia-docker exec imc python3 download.py data/validation.json data/validation
    ```
3. Training
    ```bash
    nvidia-docker exec imc python3 train.py [-h]
    ```