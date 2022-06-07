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

Capsule layer library is installed by -> pip install git+https://github.com/leftthomas/CapsuleLayer.git@master

# Run
To run the experiments just run -> python train.py

# TensorBoard logs
You can check out the tensorboard logs by running the following command in a different terminal:
tensorboard --logdir tb_logs

You will find that not much info is being logged at this moment


Hi Guys! I have written a series on articles on medium about applying Reinforcement Learning to the Multi-Stock Algorithmic Trading problem. The first article gives an easy overview of the work and subsequent ones go into implementational details. For those of you who are interested the Github Repo with full code and dataset is also made available. The Links below will take you behind the medium paywall. Clap to show some love, leave a comment if you want to share something, if you like the repo please leave a star. Cheers! 

1) Customized Deep Reinforcement Learning for Algorithmic Trading: https://medium.com/@akhileshgogikar/custom-deep-rl-for-algo-trading-106b1a2daa16?source=friends_link&sk=b06ef1dcd129be2a9dba39fa3b5c246a

2) Custom RL for Algo Trading — Data Preprocessing: https://medium.com/@akhileshgogikar/applied-rl-data-preprocessing-for-algo-trading-4478251b9676?source=friends_link&sk=e52b0d8caa6fd0b1590c8342f7b5932d

3) Custom Gym environment for multi-stock RL based Algo trading: https://medium.com/@akhileshgogikar/custom-gym-environment-for-multi-stock-algo-trading-113b07dd445d?source=friends_link&sk=e2dd8f83f3b3d3393db59fa2624d8dde

4) Customization of RL policies using StableBaselines3: https://medium.com/@akhileshgogikar/applied-rl-customizing-neural-networks-for-rl-policies-a5a9e2cf763e?source=friends_link&sk=f5415c488d17e2a5c7d9aedb2a3af181

5) Advanced deep learning customization of neural networks for RL based Algo trading: https://medium.com/@akhileshgogikar/applied-reinforcement-learning-3e73ca771bac?source=friends_link&sk=da6f6459563480f21503b37919b50e2c