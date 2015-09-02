from NeuralNetwork.neuralNetwork import *
import Training.training as training
import random
import matplotlib.pyplot as plt
from datasets.digits import *
from Logger.graphLogger import *
from Logger.consoleLogger import *
from Logger.compositeLogger import *



if __name__ == '__main__':
   training_set_X, training_set_Y = load_digits('/home/dana/Research/DeepLearning/datasets/data/digits_train.txt')
   test_set_X, test_set_Y = load_digits('/home/dana/Research/DeepLearning/datasets/data/digits_test.txt')


   # How many variables?
   numVariables = len(training_set_X[0])

   # Create the model
   NN = NeuralNetwork([numVariables, 2, 10], [None, TANH, SOFTMAX], CROSS_ENTROPY)
   NN.randomize_weights()


   graphLogger = GraphLogger(NN, (training_set_X, training_set_Y), (test_set_X, test_set_Y))
   consoleLogger = ConsoleLogger(NN, (training_set_X, training_set_Y), (test_set_X, test_set_Y))
   logger = CompositeLogger()
   logger.add_logger(graphLogger)
   logger.add_logger(consoleLogger)

   # Train the model
   training.train_batch_with_momentum(NN, training_set_X, training_set_Y, 0.8, 0.8, 0.01, 300, logger, test_set_X, test_set_Y)

   logger.log_results(NN, training_set_X, training_set_Y, test_set_X, test_set_Y)


