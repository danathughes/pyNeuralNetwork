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
##         clear.  Fix Back propagation

import numpy as np
import random


def sigmoid(z):
   """
   Sigmoid activation unit
   """

   return 1.0 / (1.0 + np.exp(-z))


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

      # Set up the weight matrices and bias terms
      # For simplicity, keep a dummy layer
      self.weights = [np.zeros((0,0))]
      self.biases = [np.zeros((0,0))]
      
      for i in range(1,len(layers)):
         self.weights.append(np.zeros( (layers[i], layers[i-1]) ))
         self.biases.append(np.zeros( (layers[i], 1) ))


      self.randomize_weights()
     

   def randomize_weights(self):
      """
      Initialize the wiehgts to small random values
      """

      # TODO: Vectorize this!
      # TODO: Allow control of the range of the random values
      # TODO: Allow control of the distribution of random values

      for i in range(1, self.numLayers):
         print self.weights[i].shape

         fanin = self.weights[i].shape[1]
         fanin = 1

         for j in range(self.weights[i].shape[0]):
            for k in range(self.weights[i].shape[1]):
                self.weights[i][j,k] = (2.0/fanin)*(random.random() - 0.5)
            self.biases[i][j,0] = (2.0/fanin)*(random.random() - 0.5)


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

      for i in range(len(dataset)):
         h = self.predict(dataset[i])
         y = np.array( [outputs[i]] ).transpose()

         # This is if I have squared error
#         cost = cost + 0.5*np.dot((y-h).transpose(), (y-h))[0,0]

         # This is if I have cross-entropy
         cost = cost - np.sum(y*np.log(h) + (1.0-y)*np.log(1.0-h))

      return cost/len(dataset)


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

         # Perform backpropagation now
         # Start with the output layer

         # This is if I have sigmoid units
#         deltas = [(target - activations[-1])*activations[-1]*(1.0-activations[-1])]

         # This is if I have softmax units
         deltas = [(target - activations[-1])]


         # Now do hidden layers
         for i in range(self.numLayers-1, 1, -1):
            delta = np.dot(self.weights[i].transpose(), deltas[0])
            delta *= activations[i-1]*(1.0 - activations[i-1])
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
         activations.append(sigmoid(self.biases[i] + np.dot(self.weights[i], activations[-1]) ))

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
      
