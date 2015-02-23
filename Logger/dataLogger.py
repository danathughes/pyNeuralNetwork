## dataLogger.py
##
## Log data to a data structure
##

import numpy as np

class DataLogger:

   def __init__(self, model, training_data=([], []), test_data=([], []), validation_data=([], [])):
      """
      """

      self.training_costs = []
      self.test_costs = []
      self.training_accuracy = []
      self.test_accuracy = []

      self.model = model
 
      self.training_data = training_data[0]
      self.training_labels = training_data[1]

      self.test_data = test_data[0]
      self.test_labels = test_data[1]

      self.validation_data = validation_data[0]
      self.validation_labels = validation_data[1]


   def log_setup(self):
      """

      """

      self.num_training_cases = len(self.training_data)
      self.num_test_cases = len(self.test_data)     


   def log_training(self, epoch_number):
      """

      """

      np.set_printoptions(precision=5)
      np.set_printoptions(suppress=True)

      correct_training_predictions = 0
      for i in range(len(self.training_data)):
         label = self.model.classify(self.training_data[i])
         if label == self.training_labels[i]:
            correct_training_predictions += 1


      correct_test_predictions = 0
      for i in range(len(self.test_data)):
         label = self.model.classify(self.test_data[i])
         if label == self.test_labels[i]:
            correct_test_predictions += 1


      self.training_costs.append(self.model.cost(self.training_data, self.training_labels))
      self.test_costs.append(self.model.cost(self.test_data, self.test_labels))
      self.training_accuracy.append(float(correct_training_predictions) / len(self.training_data))
      self.test_accuracy.append(float(correct_test_predictions) / len(self.test_data))


   def log_results(self):
      """

      """

      pass

