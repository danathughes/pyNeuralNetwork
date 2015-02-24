from NeuralNetwork.neuralNetwork import *
import Training.training as training
import random
import matplotlib.pyplot as plt
from datasets.isolet import *
from Logger.graphLogger import *
from Logger.consoleLogger import *
from Logger.compositeLogger import *
from Training.teacher import *


if __name__ == '__main__':
   training_set_X, training_set_Y = load_isolet('/home/dana/Research/DeepLearning/datasets/data/isolet_train.txt')
   test_set_X, test_set_Y = load_isolet('/home/dana/Research/DeepLearning/datasets/data/isolet_test.txt')


   # How many variables?
   numVariables = len(training_set_X[0])

   # Create the model
   NN = NeuralNetwork([numVariables, 15, 26], [None, TANH, SOFTMAX], CROSS_ENTROPY)
   NN.randomize_weights()


   graphLogger = GraphLogger(NN, (training_set_X, training_set_Y), (test_set_X, test_set_Y))
   consoleLogger = ConsoleLogger(NN, (training_set_X, training_set_Y), (test_set_X, test_set_Y))
   logger = CompositeLogger()
   logger.add_logger(graphLogger)
   logger.add_logger(consoleLogger)

   teacher = Teacher(NN, logger)
   teacher.add_weight_update(0.5, gradient_descent)
   teacher.add_weight_update(0.5, momentum)

   teacher.train_batch(training_set_X, training_set_Y, 0.001, 500)


   # Train the model

   logger.log_results(NN, training_set_X, training_set_Y, test_set_X, test_set_Y)


