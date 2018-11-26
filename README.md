# CNN-Experiment-Framework
How to run hyperparameter tuning of a CNN as an experimental design

## Introduction
This is a prototype to test the effects and interactions of changing hyperparameters. There are four factors in this experiment:

a = image_size: (64 x 64) vs (128 x 128)

b = learning rate: 0.00005 vs 0.001

c = minibatch size: 25 vs 250

d = optimizer: RMSProp vs Adam

Running `generators = fracfactgen('a b c d',3,4)` in Matlabs Statistics and Machine Learning Toolbox gives the following experimental design:

dfF = 8×4

    -1    -1    -1    -1
    
    -1    -1     1     1
    
    -1     1    -1     1
    
    -1     1     1    -1
    
     1    -1    -1     1
     
     1    -1     1    -1
     
     1     1    -1    -1
     
     1     1     1     1

Where -1 indicates low value, and +1 = high value of variable.

## Execution
Run run_experiments.py. This will build a model for each run, save the model with the name of the run, and the weights.

Tensorboard logs are created for each run, so the different runs can be compared.

Experimental analysis of results needs to be performed in MatLab.