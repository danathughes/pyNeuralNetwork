from logisticRegression import *
import training
import random
import matplotlib.pyplot as plt
from digits import *


if __name__ == '__main__':
   training_set_X, training_set_Y = load_digits('digits_train.txt')
   test_set_X, test_set_Y = load_digits('digits_test.txt')

   # How many variables?
   numVariables = len(training_set_X[0])

   # Create the model
   LR = LogisticRegressionModel(numVariables, 10)
   LR.randomize_weights()


   # Train the model
   training.train_minibatch(LR, training_set_X, training_set_Y, 0.9, 0.01, 20)


   # How'd we do?
   np.set_printoptions(precision=5)
   np.set_printoptions(suppress=True)

   print "Weights:"
   print "========"
   print LR.weights

   print
   print "Total Cost -", LR.cost(training_set_X, training_set_Y)


   print
   print "Training Results"
   print "================"
   total_training_error = 0.0
   for i in range(len(training_set_X)):
      prediction = LR.predict(training_set_X[i])
      total_training_error += abs(training_set_Y[i] - prediction)
      print "  ", i, "-", prediction, "\t", training_set_Y[i]
   mean_training_error = total_training_error / len(training_set_X)

   print
   print "Test Results"
   print "============"
   total_test_error = 0.0
   for i in range(len(test_set_X)):
      prediction = LR.predict(test_set_X[i])
      total_test_error += abs(test_set_Y[i] - prediction)
      print "  ", i, "-", prediction, "\t", test_set_Y[i]
   mean_test_error = total_test_error / len(test_set_X)

   print
   print "Total Errors"
   print "============"
   print "Total Training Error:", np.sum(total_training_error)
   print "Mean Training Error: ", np.mean(mean_training_error)
   print "Total Test Error:    ", np.sum(total_test_error)
   print "Mean Test Error:     ", np.mean(mean_test_error)

   print
   print "Training Classification Results"
   print "==============================="
   correct_predictions = 0
   for i in range(len(training_set_X)):
      classes = LR.classify(training_set_X[i])
      if classes == training_set_Y[i]:
         correct_predictions += 1
   training_classification_percentage = float(correct_predictions)/len(training_set_X)
   print "Correctly Classified:        ", correct_predictions
   print "Percent Correctly Classified:", 100*training_classification_percentage

   print
   print "Test Classification Results"
   print "==========================="
   correct_predictions = 0
   for i in range(len(test_set_X)):
      classes = LR.classify(test_set_X[i])
      if classes == test_set_Y[i]:
         correct_predictions += 1
   training_classification_percentage = float(correct_predictions)/len(test_set_X)
   print "Correctly Classified:        ", correct_predictions
   print "Percent Correctly Classified:", 100*training_classification_percentage


   


