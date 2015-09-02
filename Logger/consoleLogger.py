## consoleLogger.py
##
## Log data to the console
##

import numpy as np

class ConsoleLogger:
   """
   """

   def __init__(self, model, training_data=([], []), test_data=([], []), validation_data=([], [])):
      """
      """

      self.model = model
 
      self.training_data = training_data[0]
      self.training_labels = training_data[1]

      self.test_data = test_data[0]
      self.test_labels = test_data[1]

      self.validation_data = validation_data[0]
      self.validation_labels = validation_data[1]


   def set_training_data(self, data, labels):
      self.training_data = data
      self.training_labels = labels
      
   def set_test_data(self, data, labels):
      self.test_data = data
      self.test_labels = labels

   def log_setup(self):
      """

      """

      np.set_printoptions(precision=5)
      np.set_printoptions(suppress=True)

      print "Model"
      print "====="
      print "  Number of inputs: ", self.model.N
      print "  Number of outputs:", self.model.M
      print

      print "Training Set"
      print "============"
      print "  Number of training cases:", len(self.training_data)
      print

      print "Test Set"
      print "========"
      print "  Number of test cases:", len(self.test_data)


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

      if epoch_number % 1 == 0:
         print "Epoch #" + str(epoch_number) + ":"
         print "  Training Set Cost -", self.model.cost(self.training_data, self.training_labels)
         print "  Test Set Cost -", self.model.cost(self.test_data, self.test_labels)
         print "  Training Set Accuracy -", 100.0*correct_training_predictions/len(self.training_data) 
         print "  Test Set Accuracy -", 100.0*correct_test_predictions/len(self.test_data) 


   def log_results(self):
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

      print "Final Results:"
      print "=============="
      print "  Training Set Cost -", self.model.cost(self.training_data, self.training_labels)
      print "  Test Set Cost     -", self.model.cost(self.test_data, self.test_labels)
      print "  Training Accuracy -", correct_training_predictions, "/", len(self.training_data), "-", (100.0*correct_training_predictions)/len(self.training_data), "%"
      print "  Test Accuracy     -", correct_test_predictions, "/", len(self.test_data), "-", (100.0*correct_test_predictions)/len(self.test_data), "%"

