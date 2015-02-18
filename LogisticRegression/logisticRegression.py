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

import numpy as np
import random

## Cost and gradient functions
## TODO:  Pull these out into another namespace -- they're just functions!


def sigmoid(z):
   """
   Sigmoid activation unit
   """

   return 1.0 / (1.0 + np.exp(-z))


def gradient_sigmoid(a):
   """
   Gradient of the sigmoid unit for the given activation
   """

   return a * (1.0 - a)


def softmax(z):
   """
   Softmax activation layer
   """

   activations = np.exp(z)
   return activations / np.sum(activations)   


def gradient_softmax(a):
   """
   Gradient of the softmax unit for the given activtaion
   """

   return a * (1.0 - a)   


def cost_squared_error(predictions, targets):
   """
   Calculate the cost function of the model for the given dataset
   """

   cost = 0.0
   
   for prediction, target in zip(predictions, targets):
      cost += 0.5*(target-prediction).transpose().dot(target-prediction)[0,0]

   return cost


def gradient_squared_error(prediction, target):
   """
   Gradient of the squared error cost function w.r.t. activation
   """

   return target - prediction


def cost_cross_entropy(predictions, targets):
   """
   Calculate the cross-entropy cost
   """
   
   cost = 0.0

   for prediction, target in zip(predictions, targets):
      cost -= np.sum(target*np.log(prediction) + (1.0-target)*np.log(1.0-prediction))
   
   return cost


def gradient_cross_entropy(prediction, target):
   """
   """

   return (target - prediction) / (prediction * (1.0 - prediction))


# Create constant tuples of cost / gradient pairs for use with the Logistic 
# Regression models

SIGMOID = (sigmoid, gradient_sigmoid)
SOFTMAX = (softmax, gradient_softmax)
SQUARED_ERROR = (cost_squared_error, gradient_squared_error)
CROSS_ENTROPY = (cost_cross_entropy, gradient_cross_entropy)

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

      net = self.biases + np.dot(self.weights,np.array([data]).transpose())
      return self.activation_function(net)


   def classify(self, data):
      """
      Classify the data by assigning 1 to the highest prediction probability
      """
      
      predictions = self.predict(data)
      P_max = np.max(predictions)

      classes = [1 if p == P_max else 0 for p in predictions]
      return classes
      
