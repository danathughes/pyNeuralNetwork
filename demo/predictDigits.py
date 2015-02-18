from LogisticRegression.logisticRegression import *
import Training.training as training
import random
import matplotlib.pyplot as plt
from datasets.digits import *
import Logger.consoleLogger as Logger


if __name__ == '__main__':
   training_set_X, training_set_Y = load_digits('datasets/data/digits_train.txt')
   test_set_X, test_set_Y = load_digits('datasets/data/digits_test.txt')

   # How many variables?
   numVariables = len(training_set_X[0])

   # Create the model
   LR = LogisticRegressionModel(numVariables, 10, SOFTMAX)
   LR.randomize_weights()

   logger = Logger()

   # Train the model
   training.train_batch(LR, training_set_X, training_set_Y, 0.9, 0.01, 200, logger, test_set_X, test_set_Y)

   # Output the final results
   logger.log_results(LR, training_set_X, training_set_Y, test_set_X, test_set_Y)
