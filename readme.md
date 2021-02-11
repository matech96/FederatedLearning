# Requirements

 To run the code you will need at least one NVidia GPU.

# Installation

The code runs inside a docker container. You need to install [docker](https://docs.docker.com/get-docker/) and [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker). Once you have docker with working gpu support, you need to build the docker image.

Go to the `docker` folder and build the image!

`cd docker`

`docker build -t fedopt .`

# Run experiments

You can use the `run.py` file to run the experiments. You will need to configure it with the proper hyperparameters. Call `docker run -it --gpus 0 --env INCLUDE_TUTORIALS=false -v /path/to/FederatedLearning:/workspace -w /workspace fedopt python run.py -h` to learn more. We list the parameters for the experiments in the paper in the following table.

## Cross-device
| Epoch         | 1                                                                     | 5                                                                                        | 10                                                                               | 20                                                                               |
|---------------|-----------------------------------------------------------------------|------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| SGD-SGD       | --eval_last_20 1.0 0.31622776601683794 SGD SGD 3400 10 1.0 100        | --eval_last_20 1.0 0.1 SGD SGD 3400 10 5.0 100                                           | --eval_last_20 1.0 0.1 SGD SGD 3400 10 10.0 100                                  | --eval_last_20 1.0 0.1 SGD SGD 3400 10 20.0 100                                  |
| Yogi-SGD      | --eval_last_20 0.03162277660168379 0.1 Yogi SGD 3400 10 1.0 100       | --eval_last_20 0.03162277660168379 0.1 Yogi SGD 3400 10 5.0 100                          | --eval_last_20 0.03162277660168379 0.03162277660168379 Yogi SGD 3400 10 10.0 100 | --eval_last_20 0.03162277660168379 0.03162277660168379 Yogi SGD 3400 10 20.0 100 |
| SGD-(A)-SGDM  | --avg --eval_last_20 1.0 0.31622776601683794 SGD SGDM 3400 10 1.0 100 | --avg --eval_last_20 1.0 0.1 SGD SGDM 3400 10 5.0 100                                    | --avg --eval_last_20 1.0 0.1 SGD SGDM 3400 10 10.0 100                           | --avg --eval_last_20 1.0 0.03162277660168379 SGD SGDM 3400 10 20.0 100           |
| SGD-(A)-Yogi  | --avg --eval_last_20 1.0 0.01 SGD Yogi 3400 10 1.0 100                | --avg --eval_last_20 1.0 0.0031622776601683794 SGD Yogi 3400 10 5.0 100                  | --avg --eval_last_20 1.0 0.0031622776601683794 SGD Yogi 3400 10 10.0 100         | --avg --eval_last_20 1.0 0.01 SGD Yogi 3400 10 20.0 100                          |
| Yogi-(A)-Yogi | --avg --eval_last_20 0.01 0.01 Yogi Yogi 3400 10 1.0 100              | --avg --eval_last_20 0.03162277660168379 0.0031622776601683794 Yogi Yogi 3400 10 5.0 100 | --avg --eval_last_20 0.03162277660168379 0.001 Yogi Yogi 3400 10 10.0 100        | --avg --eval_last_20 0.03162277660168379 0.001 Yogi Yogi 3400 10 20.0 100        |

## Cross-silo

|                    | taxi                                                                     | institution                                                            |
|--------------------|--------------------------------------------------------------------------|------------------------------------------------------------------------|
| SGD-SGD            | 1.0 0.31622776601683794 SGD SGD 340 170 1.0 10                           | 1.0 0.1 SGD SGD 10 10 1.0 10                                           |
| SGD-SGD + SCAFFOLD | --scaffold 1.0 0.31622776601683794 SGD SGD 340 170 1.0 10                | --scaffold 1.0 0.1 SGD SGD 10 10 1.0 10                                |
| Yogi-SGD           | 0.1 0.1 Yogi SGD 340 170 1.0 10                                          | 0.1 0.03162277660168379 Yogi SGD 10 10 1.0 10                          |
| SGD-(A)-Yogi       | --avg 0.03162277660168379 0.00031622776601683794 SGD Yogi 340 170 1.0 10 | --avg 1.0 0.0031622776601683794 SGD Yogi 10 10 1.0 10                  |
| Yogi-(A)-Yogi      | --avg 0.1 0.0031622776601683794 Yogi Yogi 340 170 1.0 10                 | --avg 0.03162277660168379 0.0031622776601683794 Yogi Yogi 10 10 1.0 10 |
| SGD-Yogi           | 1.0 0.0031622776601683794 SGD Yogi 10 10 1.0 10                          | 1.0 0.0031622776601683794 SGD Yogi 10 10 1.0 10                        |

Type the appropiate hyperparameters at the end of the command:

`docker run -it --gpus 0 --env INCLUDE_TUTORIALS=false -v /path/to/FederatedLearning:/workspace -w /workspace fedopt python run.py <hyperparameters>`