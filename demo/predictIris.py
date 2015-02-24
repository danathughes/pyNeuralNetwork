## predictIris.py
##
## Simple script to predict iris

from LogisticRegression.logisticRegression import *
import random
#import matplotlib.pyplot as plt
import Training.training as training
from datasets.iris import *
import Preprocess.featureScaling as featureScaling
from Logger.graphLogger import *
from Logger.consoleLogger import *
from Logger.compositeLogger import *
from Training.teacher import *
from Training.crossValidation import *

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
   LR = LogisticRegressionModel(numVariables, 3, SIGMOID, CROSS_ENTROPY)
   LR.randomize_weights()

   graphLogger = GraphLogger(LR, (training_set_X, training_set_Y), (test_set_X, test_set_Y))
   consoleLogger = ConsoleLogger(LR, (training_set_X, training_set_Y), (test_set_X, test_set_Y))
   logger = CompositeLogger()
   logger.add_logger(graphLogger)
   logger.add_logger(consoleLogger)

   logger.log_setup()

   # Train the model
   teacher = Teacher(LR, logger)
   teacher.add_weight_update(0.9, gradient_descent)
   teacher.add_weight_update(0., momentum)
   teacher.add_weight_update(0.000, weight_decay)
#   teacher.train_batch(training_set_X, training_set_Y, 0.0001, 200)

   # Separate the data into 10 folds
   folds_X = [[]]*10
   folds_Y = [[]]*10

   for i in range(len(training_set_X)):
      folds_X[i%10].append(training_set_X[i])
      folds_Y[i%10].append(training_set_Y[i])

   cost, accuracy = k_fold_cross_validation(LR, teacher, folds_X, folds_Y, 0.001, 20)

   logger.log_results()

   print "Cost = ", cost
   print "Acc  = ", accuracy
