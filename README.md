# MultiStockRLTrading
Trading multiple stocks using custom gym environment and custom neural network with StableBaselines3.

This work is part of a series of articles written on medium on Applied RL:

1) Customized Deep Reinforcement Learning for Algorithmic Trading: https://medium.com/@akhileshgogikar/custom-deep-rl-for-algo-trading-106b1a2daa16

2) Custom RL for Algo Trading — Data Preprocessing: https://medium.com/@akhileshgogikar/applied-rl-data-preprocessing-for-algo-trading-4478251b9676

3) Custom Gym environment for multi-stock RL based Algo trading: https://medium.com/@akhileshgogikar/custom-gym-environment-for-multi-stock-algo-trading-113b07dd445d

4) Customization of RL policies using StableBaselines3: https://medium.com/@akhileshgogikar/applied-rl-customizing-neural-networks-for-rl-policies-a5a9e2cf763e

5) Advanced deep learning customization of neural networks for RL based Algo trading: https://medium.com/@akhileshgogikar/applied-reinforcement-learning-3e73ca771bac

# Dependencies

1) Install Conda
2) If you have a GPU install - Cuda and Cudnn to enable GPU usage 

# Installation
To install the dependencies in a conda environment by running -> conda env create -f environment.yml

# Run
To run the experiments just run -> python train.py

# TensorBoard logs
You can check out the tensorboard logs by running the following command in a different terminal:
tensorboard --logdir tb_logs

You will find that not much info is being logged at this moment
