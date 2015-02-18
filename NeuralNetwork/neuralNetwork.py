## neuralNetwork.py    Dana Hughes    version 1.02     24-November-2014
##
## Fully connected neural network model, which predicts a output vector
## given an input vector.
##
## Revisions:
##   1.0   Initial version, modified from LogisticRegressionModel with
##         batch and stochastic gradient descent.
##   1.01  Got backprop algorithm to work!  (But only for
##   1.02  Added capability to have different activations
##   1.03  Add weight decay 
##   1.04  Pull out training functionality to work with Training.training
##   1.05  Separate biases into a separate array, to make things a bit more 
##         clear.  Fix back propagation, and make it all follow the
##         logistic regression interface.
##   1.06  Create functions for cost, activiation and activation gradients
##         Abstracted activations, costs and gradients in neural network

import numpy as np
import random


# These are fine here for now, but should be moved into a separate namespace, maybe
# Functions, as they are common to many models.

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


def tanh(z):
   """
   Hyperbolic tangent activation
   """

   return (1.0 - np.exp(-z))/(1.0 + np.exp(-z))


def gradient_tanh(a):
   """
   Gradient of the hyperbolic tangent activation function
   """

   return 1.0 - a**2


def linear(z):
   """
   Simple linear activation function
   """

   return z


def gradient_linear(a):
   """
   Gradient of the linear activation function
   """

   return 1


def fast_ReLU(z):
   """
   Simple rectified linear unit
   """

   return np.max(0, z)


def gradient_fast_ReLU(a):
   """
   Gradient of the simple ReLU
   """

   gradient = a.copy()
   gradient[gradient>0] = 1
   gradient[gradient<0] = 0

   return gradient


def softplus(z):
   """
   Actual rectified linear unit / softplus
   """

   return np.log(1.0 + np.exp(z))
   

def gradient_softplus(a):
   """
   Gradient of the softplus function
   """

   return sigmoid(a)


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


# Group activations and gradients into tuples for simplicity
SIGMOID = (sigmoid, gradient_sigmoid)
SOFTMAX = (softmax, gradient_softmax)
TANH = (tanh, gradient_tanh)
RELU = (fast_ReLU, gradient_fast_ReLU)
SOFTPLUS = (softplus, gradient_softplus)
LINEAR = (linear, gradient_linear)

# Similar for cost and cost gradients
SQUARED_ERROR = (cost_squared_error, gradient_squared_error)
CROSS_ENTROPY = (cost_cross_entropy, gradient_cross_entropy)


class NeuralNetwork:
   """
   """

   def __init__(self, layers, activation_functions = None, cost_function = None):
      """
      Create a new Logistic Regression model with randomized weights
      """

      # Network layer data - number of layers, number of inputs, number of outputs
      self.N = layers[0]
      self.M = layers[-1]
      self.numLayers = len(layers)      


      # Initialize all the weights to zero.  For simplicity, keep a "dummy" layer of 
      # weights (which would be weights leading into the input layer...)
      self.weights = [np.zeros((0,0))]
      self.biases = [np.zeros((0,0))]

      for i in range(1,len(layers)):
         self.weights.append(np.zeros( (layers[i], layers[i-1]) ))
         self.biases.append(np.zeros( (layers[i], 1) ))


      # Store the activation and cost functions 
      # Default is sigmoid activation functions, and squared error cost function
      self.activation_functions = [None]*self.numLayers
      self.gradient_functions = [None]*self.numLayers

      if activation_functions == None:
         for i in range(1, self.numLayers):
            self.activation_functions[i] = SIGMOID[0]
            self.gradient_functions[i] = SIGMOID[1]
      else:
         for i in range(1,self.numLayers):
            self.activation_functions[i] = activation_functions[i][0]
            self.gradient_functions[i] = activation_functions[i][1]

      if cost_function == None:
         self.cost_function = cost_squared_error
         self.cost_gradient = gradient_squared_error
      else:
         self.cost_function = cost_function[0]
         self.cost_gradient = cost_function[1]
      
       
      # Weight symmetry *MUST* be broken.  User can set weights later if desired.
      self.randomize_weights()
     

   def randomize_weights(self):
      """
      Initialize the wiehgts to small random values
      """

      # TODO: Allow control of the range of the random values
      # TODO: Allow control of the distribution of random values

      for i in range(1, self.numLayers):

         fanin = self.weights[i].shape[1]
         fanin = 1
         
         self.weights[i] = np.array(np.random.uniform(-1/fanin, 1/fanin, self.weights[i].shape))
         self.biases[i] = np.array(np.random.uniform(-1/fanin, 1/fanin, self.biases[i].shape))


   def printNN(self):
      """
      Print NN information
      """

      print "Number of layers:", self.numLayers
      print "Number of inputs:", self.N
      print "Number of outputs:", self.M

      print

      print "Layers:"
      print "-------"

      print "Input (",self.N,") ->",

      for i in range(1, self.numLayers):
         print "Layer", i, "(", self.weights[i].shape, ") ->",

      print "Output (",self.M,")"

      print
      print "-------------------"
      print 
      
      np.set_printoptions(precision=3)
      np.set_printoptions(suppress=True)

      for i in range(1, self.numLayers):
         print "Weights - Layer",i
         print self.weights[i]
         print
         print "Biases - Layer",i
         print self.biases[i]
         print


   def cost(self, dataset, outputs):
      """
      Calculate the cost function for the given dataset
      """

      cost = 0.0

      predictions = [self.predict(data) for data in dataset]
      targets = [np.array([output]).transpose() for output in outputs]

      return self.cost_function(predictions, targets)/len(dataset)


   def gradient(self, dataset, outputs):
      """
      Determine the gradient for the weights and biases
      """

      dW = [np.zeros((0,0))]
      dB = [np.zeros((0,0))]

      # Set up for calculating the gradient of each weight
      for i in range(1, self.numLayers):
         dW.append(np.zeros(self.weights[i].shape))
         dB.append(np.zeros(self.biases[i].shape))

      for data, output in zip(dataset, outputs):
         # Convert the output to a column vector
         target = np.array([output]).transpose()

         # Do a forward pass - get the activation of each layer
         activations = self.activate(data)

         # Perform backpropagation
         # The first delta is the gradient of the cost function (dE/dz) times the gradient of the activation (dz/dy)

         deltas = [self.cost_gradient(activations[-1], target) * self.gradient_functions[-1](activations[-1])]

         # Now do hidden layers
         for i in range(self.numLayers-1, 1, -1):
            delta = np.dot(self.weights[i].transpose(), deltas[0])
            delta *= self.gradient_functions[i-1](activations[i-1])
            deltas = [delta] + deltas

         # Update the weights using the forward pass and backpropagation pass
         for i in range(1, self.numLayers):
            dW[i] -= np.dot(deltas[i-1], activations[i-1].transpose())
            dB[i] -= deltas[i-1]

      # Divide the gradient by the number of items in the dataset
      for i in range(1, self.numLayers):
         dW[i] /= len(dataset)
         dB[i] /= len(dataset)

      # All done!  Stack the dW and dB lists and return them
      return dW + dB
         


   def update_weights(self, update):
      """
      Update the weights in the model by adding dW
      """

      # Split the derivative into dW and dB
      dW = update[:self.numLayers]
      dB = update[self.numLayers:]

      for i in range(self.numLayers):
         self.weights[i] = self.weights[i] + dW[i]
         self.biases[i] = self.biases[i] + dB[i]


   def activate(self, data):
      """
      Calculate the activation of each layer
      """

      # For this version, we'll assume sigmoid activation units
      activations = [np.array( [data] ).transpose()]      # Makes it an Nx1 column vector

      for i in range(1, self.numLayers):
         # The activation of each layer is simply the activation of the output of the
         # previous layer plus the bias
         net = self.biases[i] + np.dot(self.weights[i], activations[-1])
         activations.append(self.activation_functions[i](net))

      return activations
         


   def predict(self, data):
      """
      Calculate the output layer for the given peice of data
      """

      return self.activate(data)[-1]


   def classify(self, data):
      """
      Classify the data by assigning 1 to the highest prediction probability
      """

      predictions = self.predict(data)
      P_max = np.max(predictions)

      classes = [1 if p == P_max else 0 for p in predictions]
      return classes
      
