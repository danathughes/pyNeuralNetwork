## predictIris.py
##
## Simple script to predict iris
##
## As an example, using single cases sequentially, execution time is
##
## real 0m7.183s
## user 0m6.925s
## sys  0m0.135s
##
## For comparison, batch training with full (150) dataset is
##
## real 0m0.594s
## user 0m0.415s
## sys  0m0.142s

import gnumpy as gpu
#from LogisticRegression.logisticRegression import *
from LogisticRegression.logisticRegressionGPU import *
import random
#import matplotlib.pyplot as plt
import Training.training as training
from datasets.iris import *
import Preprocess.featureScaling as featureScaling
from Logger.graphLogger import *
from Logger.consoleLogger import *
from Logger.compositeLogger import *
#from Training.teacher import *
from Training.teacherGPU import *
from Training.crossValidation import *

training_percentage = 0.8

if __name__ == '__main__':
   # Load the data

   iris_data, iris_classes = load_iris_data()

   # Normalize the input data
   iris_data = featureScaling.mean_stdev(iris_data)

   # Split into training and test data
   idx = range(iris_data.shape[0])
   random.shuffle(idx)
   numVariables = 4

   training_set_X = iris_data[idx[:120],:]
   training_set_Y = iris_classes[idx[:120],:]
   test_set_X = iris_data[idx[120:],:]
   test_set_Y = iris_classes[idx[120:],:] 

   training_set_X = gpu.garray(training_set_X)
   training_set_Y = gpu.garray(training_set_Y)
   test_set_X = gpu.garray(test_set_X)
   test_set_Y = gpu.garray(test_set_Y)

   # Create the model
   print "Creating model..."
   LR = LogisticRegressionModel(numVariables, 3, SIGMOID, CROSS_ENTROPY)
   LR.randomize_weights()

#   graphLogger = GraphLogger(LR, (training_set_X, training_set_Y), (test_set_X, test_set_Y))
#   consoleLogger = ConsoleLogger(LR, (training_set_X, training_set_Y), (test_set_X, test_set_Y))
   logger = CompositeLogger()
#   logger.add_logger(graphLogger)
#   logger.add_logger(consoleLogger)

   logger.log_setup()

   # Train the model
   teacher = Teacher(LR, logger)
   teacher.add_weight_update(0.9, gradient_descent)
   teacher.add_weight_update(0., momentum)
   teacher.add_weight_update(0.000, weight_decay)
   print "Training..."
   teacher.train_batch(training_set_X, training_set_Y, stopping_criteria)

   # Separate the data into 10 folds
#   folds_X = [[]]*10
#   folds_Y = [[]]*10

#   for i in range(len(training_set_X)):
#      folds_X[i%10].append(training_set_X[i])
#      folds_Y[i%10].append(training_set_Y[i])

#   cost, accuracy = k_fold_cross_validation(LR, teacher, folds_X, folds_Y, 0.001, 20)

   logger.log_results()

#   print "Cost = ", cost
#   print "Acc  = ", accuracy
