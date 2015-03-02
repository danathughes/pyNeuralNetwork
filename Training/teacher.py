## teacher.py    Dana Hughes    version 1.0     21-February-2015
##
## Object to handle training of arbitrary models  
##
## Revisions:
##   1.0   Initial version, modified algorithms from training.py

import numpy as np
import random


# To randomize batches for minibatch learning
def randomize_batch(data, output, numBatches):
   """
   """

   data_batches = [[]]*numBatches
   label_batches = [[]]*numBatches

   for d, l in zip(data, output):
      idx = random.randrange(0,numBatches)
      data_batches[idx].append(d)
      label_batches[idx].append(l)

   return data_batches, label_batches


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


# Temp!  This'll be used to ensure that a long history of costs are used to stop
global the_costs
the_costs = range(20)


def stopping_criteria(cost, epoch):
   """
   Decide whether or not to stop training
   """

   global the_costs

   stop = False

   stop = stop or epoch > 500
   stop = stop or cost < 0.1
   
   the_costs = the_costs[1:] + [cost]
   cost_avg = np.mean(np.array(the_costs))
   cost_std = np.std(np.array(the_costs))

   stop = stop or ((cost_std/cost_avg) < 0.025)

   stop = stop and epoch > 100

   return stop



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


   def train_batch(self, data, output, stopping = stopping_criteria):
      """
      Perform batch training using the provided data and labels
      """

      epoch = 0

      while not stopping(self.model.cost(data, output), epoch):
         if self.logger:
            self.logger.log_training(epoch)
         epoch+=1
         self.train(data, output)
   

   def train_minibatch(self, data, output, numBatches = 10, stopping = stopping_criteria):
      """
      Perform batch training using the provided data and labels
      """

      epoch = 0
      batchSize = int(len(data)/numBatches)

      while not stopping(self.model.cost(data, output), epoch):
 
         if self.logger:
            self.logger.log_training(epoch)

         epoch+=1

         data_batches, label_batches = randomize_batch(data, output, numBatches)

         for i in range(numBatches):
            batch_data = data_batches[i]
            batch_output = label_batches[i]

            self.train(batch_data, batch_output)


   def train_stochastic(self, data, output, stopping = stopping_criteria):
      """
      Perform stochastic (on-line) training using the data and labels
      """

      epoch = 0

      while not stopping(self.model.cost(data, output), epoch):
         if self.logger:
            self.logger.log_training(epoch, data, output, test_data, test_output)

         epoch+=1
         for i in range(len(data)):
            self.train([data[i]], [output[i]])

