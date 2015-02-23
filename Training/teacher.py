## teacher.py    Dana Hughes    version 1.0     21-February-2015
##
## Object to handle training of arbitrary models  
##
## Revisions:
##   1.0   Initial version, modified algorithms from training.py

import numpy as np
import random


# Weight update rules
def gradient_descent(**kwargs):
   """
   Simple update rule - return the negative of the gradient
   """

   return -kwargs['gradient']
   

def momentum(**kwargs):
   """
   Continue in the direction of the old weight update
   """

   return kwargs['old_dW']


def weight_decay(**kwargs):
   """
   Move the weights towards zero
   """
   return -kwargs['weights']


class Teacher:
   """
   """

   def __init__(self, model, logger = None):
      """
      """

      self.model = model
      self.logger = logger
      self.weight_updates = []
      self.update_rates = []
 
      self.old_dW = None


   def add_weight_update(self, update_rate, weight_update):
      """
      """

      self.weight_updates.append(weight_update)
      self.update_rates.append(update_rate)


   def train(self, data, output):
      """
      Train once on each of the items in the provided dataset
      """

      # Calculate the gradient of the cost function of the model given the data
      gradient = self.model.gradient(data, output)

      # The weight change will be the sum of every weight update provided
      dW = [np.zeros(grad.shape) for grad in gradient]
      
      # Was there a prior weight update (e.g., for momentum)?
      if self.old_dW == None:
         self.old_dW = dW

      for rate, update in zip(self.update_rates, self.weight_updates):
         for i in range(len(gradient)):
            dW[i] += rate*update(gradient=gradient[i], dw=dW[i], old_dW=self.old_dW[i], weights=self.model.get_weights()[i])


      # Update the model weights
      self.model.update_weights(dW)

      # Save the weight update
      self.old_dW = dW


   def train_batch(self, data, output, convergence = 0.0001, maxEpochs = 10000):
      """
      Perform batch training using the provided data and labels
      """

      epoch = 0

      while self.model.cost(data, output) > convergence and epoch < maxEpochs:
         if self.logger:
            self.logger.log_training(epoch)
         epoch+=1
         self.train(data, output)
      

   def train_batch_with_momentum(self, data, output, learning_rate = 0.1, momentum = 0.5, convergence = 0.0001, maxEpochs = 10000, test_data=None, test_output=None):
      """
      Perform batch training using the provided data and labels
      """

      epoch = 0

      dW = None

      while self.model.cost(data, output) > convergence and epoch < maxEpochs:
         if self.logger:
            self.logger.log_training(epoch, self.model, data, output, test_data, test_output)
         epoch+=1
         dW = train_epoch(data, output, learning_rate, dW)
         dW = [momentum*grad for grad in dW]


   def train_minibatch(self, data, output, learning_rate = 0.1, convergence = 0.0001, maxEpochs = 10000, numBatches = 10, test_data=None, test_output=None):
      """
      Perform batch training using the provided data and labels
      """

      epoch = 0
      batchSize = int(len(data)/numBatches)

      while self.model.cost(data, output) > convergence and epoch < maxEpochs:
 
         if self.logger:
            self.logger.log_training(epoch, model, data, output, test_data, test_output)

         epoch+=1

         for i in range(numBatches):
            batch_data = data[i*batchSize:(i+1)*batchSize]
            batch_output = output[i*batchSize:(i+1)*batchSize]

            self.train_epoch(batch_data, batch_output, learning_rate)


   def train_stochastic(self, data, output, learning_rate = 0.1, convergence = 0.0001, maxEpochs = 10000, test_data=None, test_output=None):
      """
      Perform stochastic (on-line) training using the data and labels
      """

      epoch = 0

      while self.model.cost(data, output) > convergence and epoch < maxEpochs:
         if self.logger:
            self.logger.log_training(epoch, data, output, test_data, test_output)

         epoch+=1
         for i in range(len(data)):
            self.train_epoch([data[i]], [output[i]])

