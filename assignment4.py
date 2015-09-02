# Assignment 4 Script

from datasets.isolet import *
from NeuralNetwork.neuralNetwork import *
from Functions.functions import *
from Training.teacher import *
from Training.crossValidation import *
from Logger.graphLogger import *
from Training.crossValidation import *

dataset, labels = load_isolet('/home/dana/Research/DeepLearning/datasets/data/isolet_train.txt')

# Split the dataset and labels into four groups of 30 for cross-validation
dataset1 = dataset[0:1560]
labels1 = labels[0:1560]

dataset2 = dataset[1560:3120]
labels2 = labels[1560:3120]

dataset3 = dataset[3120:4680]
labels3 = labels[3120:4680]

# This dataset missing the two phonemes?  Labels seem to verify that
dataset4 = dataset[4680:]
labels4 = labels[4680:]

dataset = [dataset1, dataset2, dataset3, dataset4]
labels = [labels1, labels2, labels3, labels4]

num_input = len(dataset1[0])
num_output = len(labels1[0])

num_hidden = 2

# Create a Neural network with TANH hidden layer and SOFTMAX output layer
NN = NeuralNetwork([num_input, num_hidden, num_output], [None, TANH, SOFTMAX], CROSS_ENTROPY)

graphLogger = GraphLogger(NN, (dataset1, labels1), (dataset2, labels2))

T = Teacher(NN, graphLogger)
T.add_weight_update(0.5, gradient_descent)
T.add_weight_update(0.5, momentum)
T.add_weight_update(0.001, weight_decay)

k_fold_cross_validation(NN, T, dataset, labels, graphLogger)
