# Deep Inverse Kinematics Solver
## Introduction
> The problem of Inverse Kinematics is solved with deep neural network, trained with so-far the largest human mocap database.Given the challenge of multi-solution in the problem of IK, the trained model selects the pose which is most consistent with the pose by real performer.Such consistency is validated by the comparison with the benchmark database.
## Preparation
### Hardware Requirement
Those neural networks are trained and tested on a standard computer with OS:Ubuntu, CPU:Intel® Core™ i7-6800K CPU @ 3.40GHz × 12, GPU:Nvidia TITAN Xp, Memory: 16G. Since it cost about ***60 hours*** to train the networks basing on this equipment, ***the higher configuration and larger memory*** is better. 
### Software Requirement
- Python
- Tensorflow(GPU version is better)
- Matlab
### Training Data and Testing Data
***please merge all three folder containing bvh files into one folder named TrainingSample after cloning the whole project***

run the following matlab files in folder called *TrainingDataPrepare* one by one using MATLAB
- trainingDataCrawl.m
- testingDataCrawl.m
- testingPrepare.m
## Implemention
### Training Process
run the python files in command line using the following statements 
```
>python3 fcn.py
>python3 cnn.py
>python3 rnn.py
>python3 gan.py
```
### Restore Process
run the following matlab files in each TrainingCode folders to generate new bvh file basing on training samples and testing samples
- fcnResotre.m
- cnnRestore.m
- rnnRestore.m
- ganRestore.m
