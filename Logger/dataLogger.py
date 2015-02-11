## dataLogger.py
##
## Log data to a data structure
##

import numpy as np

class DataLogger:

   def __init__(self):
      """
      """

      self.training_costs = []
      self.test_costs = []
      self.training_accuracy = []
      self.test_accuracy = []


   def log_setup(self, model, training_data, training_labels, test_data, test_labels):
      """

      """

      self.num_training_cases = len(training_data)
      self.num_test_cases = len(test_data)     


   def log_training(self, epoch_number, model, training_data, training_labels, test_data, test_labels):
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


      self.training_costs.append(model.cost(training_data, training_labels))
      self.test_costs.append(model.cost(test_data, test_labels))
      self.training_accuracy.append(float(correct_training_predictions) / len(training_data))
      self.test_accuracy.append(float(correct_test_predictions) / len(test_data))


   def log_results(self, model=None, training_data=None, training_labels=None, test_data=None, test_labels=None):
      """

      """

      pass

