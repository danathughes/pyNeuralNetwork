from NeuralNetwork.neuralNetwork import *
import Training.training as training
import random
import matplotlib.pyplot as plt
from datasets.digits import *
import Logger.dataLogger as Logger


if __name__ == '__main__':
   training_set_X, training_set_Y = load_digits('datasets/data/digits_train.txt')
   test_set_X, test_set_Y = load_digits('datasets/data/digits_test.txt')

   # How many variables?
   numVariables = len(training_set_X[0])

   # Create the model
   NN = NeuralNetwork([numVariables, 15, 10])
   NN.randomize_weights()


   logger = Logger.DataLogger()

   # Train the model
   training.train_batch_with_momentum(NN, training_set_X, training_set_Y, 0.8, 0.8, 0.01, 500, logger, test_set_X, test_set_Y)

   logger.log_results(NN, training_set_X, training_set_Y, test_set_X, test_set_Y)

   plt.figure(1)
   plt.plot(range(500), logger.training_costs, 'bo', range(500), logger.test_costs, 'r+')
   plt.show()

   plt.figure(2)
   plt.plot(range(500), logger.training_accuracy, 'bo', range(500), logger.test_accuracy, 'r+')
   plt.show()
