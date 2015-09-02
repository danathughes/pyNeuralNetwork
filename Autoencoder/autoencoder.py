## autoencoder.py    Dana Hughes    version 1.02     24-November-2014
##
## An autoencoder using the fully connected neural network model 
##
## Revisions:
##   1.0   Just a copy of neuralNetwork v1.03, since we can just 
##         use linear outputs and sigmoid hidden layers!

import numpy as np
import random

# Activation functions
SIGMOID = 'sigmoid'
TANH = 'tanh'
SOFTMAX = 'softmax'
RELU = 'relu'
LINEAR = 'linear'

class Autoencoder:
   """
   """

   def __init__(self, layers, activation_functions = None):
      """
      Create a new Autoencoder with randomized weights
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
         # TODO:  Gonna have to do some checking to make sure it's an activation
         #        function we know about!
      

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
      elif activation_function == TANH:
         return 1.0 - a**2
      elif activation_function == RELU:
         return 1.0 if a > 0 else 0
      elif activation_function == LINEAR:
         return 1.0


   def softmax(self, z):
      """
      Returns the softmax of the given vector.
      """

      results = np.zeros(len(z))

      for i in range(len(z)):
         results[i] = exp(z)

      results = results/sum(results)

      return results


   def cost(self, data, weight_decay = 0.01):
      """
      Determine the cost (error) of the parameters given the data and labels
      """

      # Add the offset term to the data
      cost = 0.0

      for i in range(len(data)):
         prediction = self.predict(data[i])

         for j in range(self.M):
            cost = cost + np.sum((prediction - data[i])**2)

      # How much are the total weights?
      total_weights = 0.0
      for i in range(self.numLayers-1):
         for j in range(self.weights[i].shape[0]):
            for k in range(1, self.weights[i].shape[1]):
               total_weights += self.weights[i][j,k]**2

      cost = cost/(2.0*len(data)) + weight_decay*total_weights/2.0

      return cost


   def train_batch(self, data, learning_rate = 0.1, convergence = 0.0001, maxEpochs = 10000, weight_decay = 0.01):
      """
      Perform batch training using the provided data and labels
      """

      epoch = 0

      while self.cost(data, weight_decay) > convergence and epoch < maxEpochs:
         print "Epoch", epoch, "- Cost:", self.cost(data, weight_decay)
         epoch+=1

         DW = [None]*(self.numLayers-1)
         for i in range(self.numLayers-1):
            DW[i] = np.zeros(self.weights[i].shape)

         for m in range(len(data)):

            # Forward propagation
            activations = self.forwardprop(data[m])

            # Backwards propagation
            deltas = self.backprop(data[m], data[m], activations)

            # Update the weights delta
            for i in range(self.numLayers-1):
               for j in range(len(activations[i])):
                  DW[i][:,j+1] += activations[i][j] * deltas[i]
               DW[i][:,0] += deltas[i]
 
         for i in range(len(self.weights)):
            decay_terms = np.zeros(self.weights[i].shape)
            decay_terms[:,1:] = weight_decay
            self.weights[i] -= learning_rate*((DW[i]/len(data)) + decay_terms)


   def train_minibatch(self, data, learning_rate = 0.1, convergence = 0.0001, maxEpochs = 10000, numBatches = 10, weight_decay = 0.01):
      """
      Perform batch training using the provided data and labels
      """

      epoch = 0

      batch_size = int(len(data)/numBatches)

      while self.cost(data, weight_decay) > convergence and epoch < maxEpochs:
         print "Epoch", epoch, "- Cost:", self.cost(data, weight_decay)
         epoch+=1

         for i in range(numBatches):
            data_batch = []
            output_batch = []

            for j in range(batch_size):
               idx = random.randrange(0,len(data))
               data_batch.append(data[idx])
  

            DW = [None]*(self.numLayers-1)
            for i in range(self.numLayers-1):
               DW[i] = np.zeros(self.weights[i].shape)

            for m in range(len(data_batch)):

               # Forward propagation
               activations = self.forwardprop(data_batch[m])

               # Backwards propagation
               deltas = self.backprop(data_batch[m], data_batch[m], activations)

               # Update the weights delta
               for i in range(self.numLayers-1):
                  for j in range(len(activations[i])):
                     DW[i][:,j+1] += activations[i][j] * deltas[i]
                  DW[i][:,0] += deltas[i]

            for i in range(len(self.weights)):
               decay_terms = np.zeros(self.weights[i].shape)
               decay_terms[:,1:] = weight_decay
               self.weights[i] -= learning_rate*((DW[i]/len(data)) + decay_terms)


   def backprop(self, data, output, activations):
      """
      Perform backwards propagation, for a single data point
      """

      deltas = [None] * self.numLayers

      # First, cacluate the error between output activation and given output

      deltas[self.numLayers-1] = -(output - activations[self.numLayers-1])
      for j in range(self.M):
         deltas[self.numLayers-1][j] *= self.d_activate(activations[self.numLayers-1][j], self.activation_functions[self.numLayers-1])

      for i in range(self.numLayers-2, 0, -1):
         deltas[i] = np.zeros(self.weights[i].shape[1]-1)
         for j in range(self.weights[i].shape[1] - 1):
            deltas[i][j] = np.sum(self.weights[i][:,j+1] * deltas[i+1])
            deltas[i][j] *= self.d_activate(activations[i][j], self.activation_functions[i])

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
            activation[j] = self.activate(activation[j], self.activation_functions[i+1])

         activations.append(activation)
         x = activation

      return activations


   def predict(self,data):
      """
      """

      activations = self.forwardprop(data)
      return activations[-1]


   def encode(self, data, code_level = 1):
      """
      """

      activations = self.forwardprop(data)
      return np.array(activations[code_level])
