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

import numpy as np
import random

# Activation functions
SOFTMAX = 'softmax'
SIGMOID = 'sigmoid'
TANH = 'tanh'
SOFTMAX = 'softmax'
RELU = 'relu'
LINEAR = 'linear'


## Cost and gradient functions

def cost_sigmoid(model, dataset, outputs):
   """
   Cost function for sigmoid output units
   """

   # Add the offset term to the data
   cost = 0.0

   for i in range(len(dataset)):
      prediction = model.predict(dataset[i])

      for j in range(model.M):
         cost = cost - ((1.0-outputs[i][j])*np.log(1.0-prediction[j]) + outputs[i][j]*np.log(prediction[j]))

   return cost/len(dataset)


def cost_softmax(model, dataset, outputs):
   """
   Cost function for softmax output units (i.e., cross-entropy error
   """

   cost = 0.0

   for i in range(len(dataset)):
      prediction = model.predict(dataset[i])

      for j in range(model.M):
         cost = cost - outputs[i][j]*np.log(prediction[j])

   return cost/len(dataset)



def gradient_sigmoid(model, dataset, outputs):
   """
   Gradient for sigmoid output units
   """

   gradient = np.zeros((model.N + 1, model.M)) 

   for k in range(len(dataset)):
      prediction = model.predict(dataset[k])

      for j in range(model.M):
         gradient[0,j] -= prediction[j]*(1.0 - prediction[j])*(outputs[k][j] - prediction[j])

         for i in range(model.N):
            gradient[i+1,j] -= dataset[k][i]*prediction[j]*(1.0 - prediction[j])*(outputs[k][j] - prediction[j])
  
   return gradient/len(dataset)


def gradient_softmax(model, dataset, outputs):
   """
   Gradient for softmax output units
   """

   gradient = np.zeros((model.N + 1, model.M)) 

   for k in range(len(dataset)):
      prediction = model.predict(dataset[k])

      for j in range(model.M):
         gradient[0,j] -= (outputs[k][j] - prediction[j])

         for i in range(model.N):
            gradient[i+1,j] -= dataset[k][i]*(outputs[k][j] - prediction[j])
  
   return gradient/len(dataset)


def activate_sigmoid(model, data):
   """
   Provide a prediction for the data provided
   """

   prediction = np.zeros(model.M)

   for i in range(model.M):         
      prediction[i] = model.sigmoid(model.weights[0,i] + np.sum(model.weights[1:,i]*np.array(data)))

   return prediction


def activate_softmax(model, data):
   """
   Provide a prediction for the data provided
   """

   prediction = np.zeros(model.M)

   for i in range(model.M):         
      prediction[i] = np.exp(model.weights[0,i] + np.sum(model.weights[1:,i]*np.array(data)))

   partition = sum(prediction)

   for i in range(model.M):
      prediction[i] = prediction[i]/partition

   return prediction



class NeuralNetwork:
   """
   """

   def __init__(self, layers, activation_functions = None):
      """
      Create a new Logistic Regression model with randomized weights
      """

      # Set the weights to zero, including an extra weight as the offset
      self.N = layers[0]
      self.M = layers[-1]
      self.numLayers = len(layers)

      # Keep track of the activation functions of each layer.  Note that the 
      # first layer should remain None (as it's input)
      self.activation_functions = [None]*self.numLayers

      # Did the user specify?  If not, simply assume sigmoid
      if not activation_functions:
         for i in range(1,self.numLayers):
            self.activation_functions[i] = SIGMOID
      else:
         self.activation_functions = activation_functions
      

      # Set up the weight matrices
      self.weights = []
      
      for i in range(1,len(layers)):
         self.weights.append(np.zeros((layers[i], layers[i-1] + 1)))

      # Initialize the weights to small random values
      for i in range(len(self.weights)):
         for j in range(self.weights[i].shape[0]):
            for k in range(self.weights[i].shape[1]):
               self.weights[i][j,k] = 0.02*random.random() - 0.01
     

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

      for i in range(self.numLayers-1):
         print "Layer", i, "(", self.weights[i].shape, ") ->",

      print "Output (",self.M,")"

      print
      print "-------------------"
      print 
      
      np.set_printoptions(precision=3)
      np.set_printoptions(suppress=True)

      for i in range(self.numLayers-1):
         print "Weights - Layer",i
         print self.weights[i]
         print


   def activate(self, z, activation_function):
      """
      Perform the activation function of the neurons on the value
      """

      if activation_function == SIGMOID:
         return 1.0/(1.0 + np.exp(-z))
      elif activation_function == SOFTMAX:
         activity = np.exp(-z)
         partition = np.sum(activity)
         return activity/partition
      elif activation_function == TANH:
         return (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))
      elif activation_function == RELU:
         return z if z > 0 else 0
      elif activation_function == LINEAR:
         return z

   
   def d_activate(self, a, activation_function):
      """
      Return the derivative of the activated neuron.  Note that the passed 
      value should have already been through the activation function
      """

      if activation_function == SIGMOID:
         return a*(1.0 - a)
      elif activation_function == SOFTMAX:
         return -1.0
      elif activation_function == TANH:
         return 1.0 - a**2
      elif activation_function == RELU:
         return 1.0 if a > 0 else 0
      elif activation_function == LINEAR:
         return 1.0


   def cost(self, data, output):
      """
      Determine the cost (error) of the parameters given the data and labels
      """

      # Add the offset term to the data
      cost = 0.0

      for i in range(len(data)):
         prediction = self.predict(data[i])

         for j in range(self.M):
            cost = cost + np.sum((prediction-output[i])**2)

      cost = cost/len(data)

      return cost


   def gradient(self, data, output):
      """
      Get the gradient of the error function with respect to the weights
      """

      DW = [None]*(self.numLayers-1)
      for i in range(self.numLayers-1):
         DW[i] = np.zeros(self.weights[i].shape)

      for m in range(len(data)):

         # Forward propagation
         activations = self.forwardprop(data[m])

         # Backwards propagation
         deltas = self.backprop(data[m], output[m], activations)


         # Update the weights delta
         for i in range(self.numLayers-1):
            for j in range(len(activations[i])):
               DW[i][:,j+1] += activations[i][j] * deltas[i]
            DW[i][:,0] += deltas[i]

      for i in range(self.numLayers-1):
         DW[i] = DW[i]/len(data)
 
      return DW


   def update_weights(self, dW):
      """
      Update all the weights
      """

      for i in range(self.numLayers - 1):
         self.weights[i] += dW[i]


   def backprop(self, data, output, activations):
      """
      Perform backwards propagation, for a single data point
      """

      deltas = [None] * self.numLayers

      # First, cacluate the error between output activation and given output
      # Is it softmax?   Then we have a different thing to do
      deltas[self.numLayers-1] = -(output - activations[self.numLayers-1])
      deltas[self.numLayers-1] *= self.d_activate(activations[self.numLayers-1], self.activation_functions[self.numLayers-1])


      for i in range(self.numLayers-2, 0, -1):
         deltas[i] = np.zeros(self.weights[i].shape[1]-1)
         for j in range(self.weights[i].shape[1] - 1):
            deltas[i][j] = np.sum(self.weights[i][:,j+1] * deltas[i+1])
         deltas[i] *= self.d_activate(activations[i], self.activation_functions[i])

      return deltas[1:]


   def forwardprop(self, data):
      """
      Perform forward propagation, giving the activations at each layer
      """

      x = np.array(data)
      activations = [x]

      # Calculate the activation of each layer in the neural network
      for i in range(self.numLayers-1):
         activation = np.zeros(self.weights[i].shape[0])
         for j in range(self.weights[i].shape[0]):
            activation[j] = np.sum(self.weights[i][j,0] + self.weights[i][j,1:] * x)
         activation = self.activate(activation, self.activation_functions[i+1])

         activations.append(activation)
         x = activation

      return activations


   def predict(self,data):
      """
      """

      activations = self.forwardprop(data)
      return activations[-1]

   
   def classify(self, data):
      """
      Classify the data by assigning 1 to the highest prediction probability
      """
      
      predictions = self.predict(data)
      P_max = np.max(predictions)

      classes = [1 if p == P_max else 0 for p in predictions]
      return classes
