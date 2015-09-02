## logisticRegression.py    Dana Hughes    version 1.0     24-November-2014
##
## Logistic Regression model, which predicts a single output value given an 
## input vector.
##
## Revisions:
##   1.0   Initial version, modified from LinearRegressionModel with
##         batch and stochastic gradient descent.
##   1.01  Modified output to be softmax vector
##   1.02  Fixed the predictor.  Converges much better now!
##   1.03  Separated training algorithms into a separate file
##         Set up model to accept cost / gradient functions, rather than
##         modifying specific implementations
##         Added cost / gradient functions for sigmoid and softmax layers
##   1.04  Changed from one large weight matrix to weights and biases
##         Vectorized weight randomization function
##   1.05  Vectorized everything.  Matches neural network code.
##   1.1   Major change - require input / output pairs to be 2D numpy 
##         arrays, so batch learning is performed instead of individual
##         training cases, (but still allow individual training cases)

import numpy as np
import random
from Functions.functions import *


class LogisticRegressionModel:
   """
   """

   def __init__(self, numVariables, numOutputs, activation_function = SOFTMAX, cost_function = CROSS_ENTROPY):
      """
      Create a new Logistic Regression model with randomized weights
      """

      # Set the weights to zero, including an extra weight as the offset
      self.N = numVariables
      self.M = numOutputs
      self.weights = np.zeros((self.M, self.N))     
      self.biases = np.zeros((self.M, 1))
      self.cost_function = cost_function[0]
      self.cost_gradient = cost_function[1]
      self.activation_function = activation_function[0]
      self.activation_gradient = activation_function[1]


   def randomize_weights(self):
      """
      Set all the weights to a value in the range of 1/fan_in
      """

      self.weights = np.random.uniform(-1.0/self.N, 1.0/self.N, self.weights.shape)
      self.biases = np.random.uniform(-1.0/self.N, 1.0/self.N, self.biases.shape)


   def cost(self, dataset, outputs):
      """
      Determine the cost (error) of the parameters given the data and labels
      """

      predictions = [self.predict( data ) for data in dataset]
      targets = [np.array([output]).transpose() for output in outputs]

      return self.cost_function(predictions, targets)/len(dataset)


   def gradient(self, dataset, outputs):
      """
      Determine the gradient of the parameters given the data and labels

      Gradient for the sigmoid output is dy/dx = x*y*(1-y)
      """

      dW = np.zeros((self.M, self.N)) 
      dB = np.zeros((self.M, 1))

      for data, output in zip(dataset, outputs):
         prediction = self.predict(data)
         target = np.array([output]).transpose()

         cost_gradient = self.cost_gradient(prediction, target)
         activation_gradient = self.activation_gradient(prediction)

         gradient = cost_gradient * activation_gradient
      
         dB -= gradient
         dW -= np.dot(gradient, np.array([data]))

      return [dW/len(dataset), dB/len(dataset)]


   def get_weights(self):
      """
      Provide the weights and biases
      """

      return [self.weights, self.biases]


   def update_weights(self, dW):
      """
      Update the weights in the model by adding dW
      """

      # There's only one set of weights for regression model
      self.weights += dW[0]
      self.biases += dW[1]


   def predict(self, data):
      """
      Predict the class probabilites given the data
      """

      net = self.biases + np.dot(self.weights, data.transpose())
      return self.activation_function(net)


   def classify(self, data):
      """
      Classify the data by assigning 1 to the highest prediction probability
      """
      
      predictions = self.predict(data)
      P_max = np.max(predictions)

      classes = [1 if p == P_max else 0 for p in predictions]
      return classes
      
