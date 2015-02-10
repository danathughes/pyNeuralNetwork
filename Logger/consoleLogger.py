## consoleLogger.py
##
## Log data to the console
##

import numpy as np

def log_setup(model, training_data, training_labels, test_data, test_labels):
   """

   """

   np.set_printoptions(precision=5)
   np.set_printoptions(suppress=True)

   print "Model"
   print "====="
   print "  Number of inputs: ", model.N
   print "  Number of outputs:", model.M
   print

   print "Training Set"
   print "============"
   print "  Number of training cases:", len(training_data)
   print

   print "Test Set"
   print "========"
   print "  Number of test cases:", len(test_data)


def log_training(epoch_number, model, training_data, training_labels, test_data, test_labels):
   """

   """

   np.set_printoptions(precision=5)
   np.set_printoptions(suppress=True)

   correct_training_predictions = 0
   for i in range(len(training_data)):
      classes = model.classify(training_data[i])
      if classes == training_labels[i]:
         correct_training_predictions += 1


   correct_test_predictions = 0
   for i in range(len(test_data)):
      classes = model.classify(test_data[i])
      if classes == test_labels[i]:
         correct_test_predictions += 1


   print "Epoch #" + str(epoch_number) + ":"
   print "  Training Set Cost -", model.cost(training_data, training_labels)
   print "  Test Set Cost     -", model.cost(test_data, test_labels)
   print "  Training Accuracy -", correct_training_predictions, "/", len(training_data), "-", (100.0*correct_training_predictions)/len(training_data), "%"
   print "  Test Accuracy     -", correct_test_predictions, "/", len(test_data), "-", (100.0*correct_test_predictions)/len(test_data), "%"


def log_results(model=None, training_data=None, training_labels=None, test_data=None, test_labels=None):
   """

   """

   np.set_printoptions(precision=5)
   np.set_printoptions(suppress=True)

