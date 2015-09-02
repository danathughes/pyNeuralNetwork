## functions.py    Dana Hughes    version 1.00	    17-February-2015
##
## Common functions used for activation and cost
##
## Revisions:
##   1.0   Initial version.
##   1.1   Designed to work with 2D arrays of data.  Each row of the data array 
##         is an individual sample


import numpy as np


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
   Softmax activation layer.  
   """

   activations = np.exp(z)
   # Add row-wise the activations to normalize
   denoms = np.array([np.sum(activations,1)]).transpose()
   return activations / denoms


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

   cost = 0.5*np.sum((targets-predictions)**2)

   return cost


def gradient_squared_error(predictions, targets):
   """
   Gradient of the squared error cost function w.r.t. activation
   """

   return targets - predictions


def cost_cross_entropy(predictions, targets):
   """
   Calculate the cross-entropy cost
   """
   
   cost = -np.sum(targets*np.log(predictions + 0.0001) + (1.0 - targets)*np.log(1.0 - predictions + 0.0001))

   return cost


def gradient_cross_entropy(predictions, targets):
   """
   """

   return (targets - predictions) / (0.0001 + predictions * (1.0 - predictions))


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

