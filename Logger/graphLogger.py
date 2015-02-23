## graphLogger.py
##
## Log data to a graph
##

import numpy as np
import matplotlib.pyplot as plt

class GraphLogger:
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


      self.training_costs = []
      self.test_costs = []
      self.training_accuracy = []
      self.test_accuracy = []

      # Set up figures to plot the error and accuracy
      self.fig = plt.figure(1)
      self.fig.hold(True)
      self.fig.show()

   def log_setup(self):
      """

      """

      pass


   def log_training(self, epoch_number):
      """

      """

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
      self.training_accuracy.append(100.0*correct_training_predictions/len(self.training_data))
      self.test_accuracy.append(100.0*correct_test_predictions/len(self.test_data))

      # Plot the data!
      plt.subplot(121)
      xaxis = range(len(self.training_costs))
      plt.plot(xaxis, self.training_costs, '-b', xaxis, self.test_costs, '-r')
      plt.axis([0, len(xaxis), 0, np.max(self.training_costs + self.test_costs)])
      plt.xlabel('Epoch Number')
      plt.ylabel('Cost')
      plt.title('Cost vs. Epoch')
      plt.legend(['Training Set', 'Test Set'], 'upper right')

      plt.subplot(122)
      plt.plot(xaxis, self.training_accuracy, '-b', xaxis, self.test_accuracy, '-r')
      plt.axis([0, len(xaxis), 0, 100])
      plt.xlabel('Epoch Number')
      plt.ylabel('Accuracy')
      plt.title('Accuracy vs. Epoch')
      plt.legend(['Training Set', 'Test Set'], 'lower right')



      plt.draw()
      self.fig.show()
 


   def log_results(self):
      """

      """

      plt.show()
      self.fig.show()
