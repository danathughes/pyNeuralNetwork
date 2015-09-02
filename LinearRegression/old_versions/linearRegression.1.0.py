## linearRegression.py    Dana Hughes    version 1.0     17-November-2014
##
## Linear Regression model, which predicts a single output value given an 
## input vector.
##
## Revisions:
##   1.0   Initial version, consisting of a LinearRegressionModel with
##         batch and stochastic gradient descent.

import numpy as np
import random
import matplotlib.pyplot as plt

d = [[1.1,-1.1],[2.1,-1.7],[0.0,-0.3]]
o = [1.0,2.0,3.0]

class LinearRegressionModel:
   """
   """

   def __init__(self, numVariables):
      """
      """

      # Set the weights to zero, including an extra weight as the offset
      self.N = numVariables
      self.weights = np.zeros((numVariables + 1,1))
     

   def cost(self, data, output):
      """
      """

      # Add the offset term to the data
      cost = 0.0

      for i in range(len(data)):
         prediction = self.predict(data[i])
         error = prediction - output[i]
         cost = cost + error**2

      return cost/2


   def gradient(self, data, output):
      """
      """

      gradient = [0] * (self.N + 1)

      errors = []
      for i in range(len(data)):
         prediction = self.predict(data[i])
         errors.append(output[i] - prediction)

      gradient[0] = sum(errors)

      for j in range(self.N):
         gradient[j+1] = 0
         for i in range(len(errors)):
            gradient[j+1] = gradient[j+1] + data[i][j] * errors[i]

      return gradient


   def train_batch(self, data, output, learning_rate = 0.1, convergence = 0.0001, maxEpochs = 10000):
      """
      """

      epoch = 0

      while self.cost(data, output) > convergence and epoch < maxEpochs:
         print "Epoch", epoch, "- Cost:", self.cost(data,output)
         epoch+=1
         gradient = self.gradient(data, output)
         for i in range(self.N + 1):
            self.weights[i] += learning_rate * gradient[i]
            

   def train_stochastic(self, data, output, learning_rate = 0.1, convergence = 0.0001, maxEpochs = 10000):
      """
      """

      epoch = 0

      while self.cost(data, output) > convergence and epoch < maxEpochs:
         print "Epoch", epoch, "- Cost:", self.cost(data,output)
         epoch+=1
         for i in range(len(data)):
#            gradient = self.gradient([data[i]], [output[i]])
            error = output[i] - self.predict(data[i])
            self.weights[0] += learning_rate * error
            for j in range(self.N):
               self.weights[j+1] += learning_rate * error * data[i][j]
 


   def predict(self, data):
      """
      """

      # Start with the offset term
      prediction = self.weights[0]

      for i in range(self.N):
         prediction = prediction + self.weights[i+1] * data[i]

      return prediction
