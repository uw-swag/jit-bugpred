# JIT Bug Prediction

This repository contains the scripts for the paper 
***JITGNN: A Deep Graph Neural Network
Framework for Just-In-Time Bug Prediction***. These scripts included scripts for data processing, training, and testing the JITGNN model.


Here is the guide to the scripts under directory `src/`:

* `main.py` is the main runner to train and test the model.
* `models.py` contains the implementation of the JITGNN model.
* `train.py` is the script that has both the training and testing functions.
* `datasets.py` is the Pytorch data handler script that prepares the samples in the dataset and feed them to the training model.
* `gumtree.py` is the script that obtains the ASTs of the source codes.