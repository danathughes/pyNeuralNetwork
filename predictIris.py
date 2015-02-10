## predictIris.py
##
## Simple script to predict iris

from LogisticRegression.logisticRegression import *
import random
import matplotlib.pyplot as plt
import Training.training as training
from datasets.iris import *
import Preprocess.featureScaling as featureScaling
import Logger.consoleLogger as logger

training_percentage = 0.8

if __name__ == '__main__':
   # Load the data

   iris_data, iris_classes = load_iris_data()

   # Normalize the input data
   iris_data = featureScaling.mean_stdev(iris_data)

   # Split into training and test data
   training_set_X = []
   training_set_Y = []
   test_set_X = []
   test_set_Y = []

   for i in range(len(iris_data)):
      if random.random() < training_percentage:
         training_set_X.append(iris_data[i])
         training_set_Y.append(iris_classes[i])
      else:
         test_set_X.append(iris_data[i])
         test_set_Y.append(iris_classes[i])

   # How many variables?
   numVariables = len(training_set_X[0])

   # Create the model
   LR = LogisticRegressionModel(numVariables, 3, SOFTMAX)
   LR.randomize_weights()

   logger.log_setup(LR, training_set_X, training_set_Y, test_set_X, test_set_Y)

   # Train the model
   training.train_batch(LR, training_set_X, training_set_Y, 0.5, 0.0001, 200, logger, test_set_X, test_set_Y)



