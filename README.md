# IMaterialist classifier

## Description

## Prerequisites

You will need the following things properly installed on your computer.

* [Docker](https://www.docker.com/)

## Installation

* `git clone `

## Running

Remember that Docker container has the Python version 2.7.12!

1. Download data
1. Download pretrained .
2. If you are planning to use nvidia-docker, you need to building nvidia-docker image first. Otherwise, you can skip this step
    ```bash
    nvidia-docker build -t sm_keras_tf_py3:gpu .
    ```
    Run container
    ```bash
    nvidia-docker run -v $PWD/src:/imaterialist -dt --name imc sm_keras_tf_py3:gpu /bin/bash
    ```
3. Training
    ```bash
    nvidia-docker exec tcc python train.py [-h]
    ```