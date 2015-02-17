## training.py    Dana Hughes    version 1.0     09-February-2015
##
## Functions used to train arbitrary models.  
##
## Revisions:
##   1.0   Initial version, modified algorithms from LogisticRegression

import numpy as np
import random


def train_epoch(model, data, output, learning_rate = 0.1, momentum = None):
   """
   Train once on each of the items in the provided dataset
   """

   gradient = np.array(model.gradient(data, output))
   if momentum == None:
      dW = [-learning_rate * grad for grad in gradient]
   else:
      dW = [-learning_rate * grad + grad_momentum for grad,grad_momentum in zip(gradient, momentum)]
   model.update_weights(dW)

   return dW


def train_batch(model, data, output, learning_rate = 0.1, convergence = 0.0001, maxEpochs = 10000, logger=None, test_data=None, test_output=None):
   """
   Perform batch training using the provided data and labels
   """

   epoch = 0

   while model.cost(data, output) > convergence and epoch < maxEpochs:
      if logger:
         logger.log_training(epoch, model, data, output, test_data, test_output)
      epoch+=1
      train_epoch(model, data, output, learning_rate)
      

def train_batch_with_momentum(model, data, output, learning_rate = 0.1, momentum = 0.5, convergence = 0.0001, maxEpochs = 10000, logger=None, test_data=None, test_output=None):
   """
   Perform batch training using the provided data and labels
   """

   epoch = 0

   dW = None

   while model.cost(data, output) > convergence and epoch < maxEpochs:
      if logger:
         logger.log_training(epoch, model, data, output, test_data, test_output)
      epoch+=1
      dW = train_epoch(model, data, output, learning_rate, dW)
      dW = [momentum*grad for grad in dW]

def train_minibatch(model, data, output, learning_rate = 0.1, convergence = 0.0001, maxEpochs = 10000, numBatches = 10, logger=None, test_data=None, test_output=None):
   """
   Perform batch training using the provided data and labels
   """

   epoch = 0
   batchSize = int(len(data)/numBatches)

   while model.cost(data, output) > convergence and epoch < maxEpochs:

      if logger:
         logger.log_training(epoch, model, data, output, test_data, test_output)

      epoch+=1

      for i in range(numBatches):
         batch_data = data[i*batchSize:(i+1)*batchSize]
         batch_output = output[i*batchSize:(i+1)*batchSize]

         train_epoch(model, batch_data, batch_output, learning_rate)


def train_stochastic(model, data, output, learning_rate = 0.1, convergence = 0.0001, maxEpochs = 10000, logger=None, test_data=None, test_output=None):
   """
   Perform stochastic (on-line) training using the data and labels
   """

   epoch = 0

   while model.cost(data, output) > convergence and epoch < maxEpochs:
      if logger:
         logger.log_training(epoch, model, data, output, test_data, test_output)

      epoch+=1
      for i in range(len(data)):
         train_epoch(model, [data[i]], [output[i]])

