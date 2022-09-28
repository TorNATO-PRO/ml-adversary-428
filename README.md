# MLAdversary

This is the repository containing the code of the ML adversary project for Cpts 428.

In order to run this repository, you will want to follow the instructions to install Tensorflow's Docker container [here](https://www.tensorflow.org/install/docker). 

Then, you will want to run the environment setup script!

`$ python env_setup.py`

Afterwards, you will want to build the Docker container as follows:

`$ docker build -t ml_adversary .`

Then, you can access the environment as follows:

```
$ sudo docker run \ 
       --gpus all \
       --rm \ 
       -it \
       --name ml_adversary_container \
       -v "$(pwd)"/output,target=/home/ml_adversary/output \
       -v "$(pwd)"/saved_models,target=/home/ml_adversary/saved_models \
       ml_adversary bash
```
