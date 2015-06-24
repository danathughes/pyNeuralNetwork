## neuralNetwork.py    Dana Hughes    version 1.02     24-November-2014
##
## Fully connected neural network model, which predicts a output vector
## given an input vector.
##
## Revisions:
##   1.0   Initial version, copyed from NeuralNetwork.py
##         For simplicity, we'll keep only a single hidden layer

import numpy as np
import random
from Functions.functions import *


class RecurrentNeuralNetwork:
   """
   """

   def __init__(self, layers, activation_functions = None, cost_function = None):
      """
      Create a new RNN model with randomized weights
      """

      # Network layer data - number of layers, number of inputs, number of outputs
      self.N = layers[0]
      self.M = layers[-1]
      self.numLayers = len(layers)      


      # Initialize all the weights to zero.  For simplicity, keep a "dummy" layer of 
      # weights (which would be weights leading into the input layer...)
      self.Wih = np.zeros((layers[1], layers[0]))
      self.Whh = np.zeros((layers[1], layers[1]))
      self.Who = np.zeros((layers[2], layers[1]))

      self.bh = np.zeros((layers[1], 1))
      self.bo = np.zeros((layers[2], 1))

#      self.weights = [np.zeros((0,0))]
#      self.recurrent_weights = [np.zeros((0,0))]
#      self.biases = [np.zeros((0,0))]

#      for i in range(1,len(layers)):
#         self.weights.append(np.zeros( (layers[i], layers[i-1]) ))
#         self.recurrent_weights.append(np.zeros( (layers[i], layers[i]) ))
#         self.biases.append(np.zeros( (layers[i], 1) ))


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

#      for i in range(1, self.numLayers):

#         fanin = self.weights[i].shape[1]

      self.Wih = np.array(np.random.uniform(-1.0, 1.0, self.Wih.shape))
      self.Whh = np.array(np.random.uniform(-1.0, 1.0, self.Whh.shape))
      self.Who = np.array(np.random.uniform(-1.0, 1.0, self.Who.shape))

      self.bh = np.array(np.random.uniform(-1.0, 1.0, self.bh.shape))
      self.bo = np.array(np.random.uniform(-1.0, 1.0, self.bo.shape))
          
#         self.weights[i] = np.array(np.random.uniform(-1.0, 1.0, self.weights[i].shape))
#         self.recurrent_weights[i] = np.array(np.random.uniform(-1.0, 1.0, self.recurrent_weights[i].shape))
#         self.biases[i] = np.array(np.random.uniform(-1.0, 1.0, self.biases[i].shape))

      self.Wih = self.Wih * 2.0 / np.sum(np.abs(self.Wih))
      self.Whh = self.Whh * 2.0 / np.sum(np.abs(self.Whh))
      self.Who = self.Who * 2.0 / np.sum(np.abs(self.Who))

      self.bh = self.bh * 2.0 / np.sum(np.abs(self.bh))
      self.bo = self.bo * 2.0 / np.sum(np.abs(self.bo))

#         self.weights[i] = self.weights[i] * 2.0 / np.sum(np.abs(self.weights[i]))
#         self.recurrent_weights[i] = self.recurrent_weights[i] * 2.0 / np.sum(np.abs(self.recurrent_weights[i]))
#         self.biases[i] = self.biases[i] * 2.0 / np.sum(np.abs(self.biases[i]))


   def cost(self, sequence_set, output_sequences):
      """
      Calculate the cost function for the given dataset
      """

      cost = 0.0
      
      total_steps = 0

      for i in range(len(sequence_set)):
         predictions = self.predict(sequence_set[i])
         for j in range(len(predictions)):
            total_steps += 1
            cost += 0.5 * (output_sequences[i][j] - predictions[j])**2

      return np.sum(cost) / total_steps

#      predictions = [self.predict(data) for data in dataset]
#      targets = [np.array([output]).transpose() for output in outputs]

#      return self.cost_function(predictions, targets)/len(dataset)


   def gradient(self, sequence_set, output_sequences):
      """
      Determine the gradient for the weights and biases
      """

      dWih = np.zeros(self.Wih.shape)
      dWhh = np.zeros(self.Whh.shape)
      dWho = np.zeros(self.Who.shape)

      dbh = np.zeros(self.bh.shape)
      dbo = np.zeros(self.bo.shape)

      # How many unrollings are there?
      frame_count = 0

#      dW = [np.zeros((0,0))]
#      dB = [np.zeros((0,0))]

      # Set up for calculating the gradient of each weight
#      for i in range(1, self.numLayers):
#         dW.append(np.zeros(self.weights[i].shape))
#         dB.append(np.zeros(self.biases[i].shape))  

      for sequence, output in zip(sequence_set, output_sequences):
         activations = self.activate(sequence)

#      for data, output in zip(dataset, outputs):
         # Convert the output to a column vector
#         target = np.array([output]).transpose()

         # Do a forward pass - get the activation of each layer
#         activations = self.activate(data)

         # Perform backpropagation
         # The first delta is the gradient of the cost function (dE/dz) times the gradient of the activation (dz/dy)

#         deltas = [self.cost_gradient(activations[-1], target) * self.gradient_functions[-1](activations[-1])]

         old_hidden = np.zeros(self.bh.shape)

         for t in range(len(activations)):
            act = activations[t]
            out = np.array(output[t])        # Why do I need to cast this?
            # Delta rule goes here...
            delta_out = self.cost_gradient(act[2], out)
            delta_out *= self.gradient_functions[2](act[2])

            delta_hidden = np.dot(self.Who.transpose(), delta_out) 
            delta_hidden *= self.gradient_functions[1](act[1])

            dWih += np.dot(delta_hidden, act[0].transpose())
            dWhh += np.dot(delta_hidden, old_hidden.transpose())
            dWho += np.dot(delta_out, act[1].transpose())

            dbh += delta_hidden
            dbo += delta_out

            frame_count += 1
            old_hidden = act[1]

         # Now do hidden layers
#         for i in range(self.numLayers-1, 1, -1):
#            delta = np.dot(self.weights[i].transpose(), deltas[0])
#            delta *= self.gradient_functions[i-1](activations[i-1])
#            deltas = [delta] + deltas

         # Update the weights using the forward pass and backpropagation pass
#         for i in range(1, self.numLayers):
#            dW[i] -= np.dot(deltas[i-1], activations[i-1].transpose())
#            dB[i] -= deltas[i-1]

      # Divide the gradient by the number of items in the dataset
#      for i in range(1, self.numLayers):
#         dW[i] /= len(dataset)
#         dB[i] /= len(dataset)

      dWih /= frame_count
      dWhh /= frame_count
      dWho /= frame_count
      dbh /= frame_count
      dbo /= frame_count

      # All done!  Stack the dW and dB lists and return them
#      return dW + dB
      
      return [dWih, dWhh, dWho, dbh, dbo]


   def get_weights(self):
      """
      Provide the weights and biases
      """

#      return self.weights + self.biases
      return [self.Wih, self.Whh, self.Who, self.bh, self.bo]


   def update_weights(self, update):
      """
      Update the weights in the model by adding dW
      """

      # Split the derivative into dW and dB
#      dW = update[:self.numLayers]
#      dB = update[self.numLayers:]

#      for i in range(self.numLayers):
#         self.weights[i] = self.weights[i] + dW[i]
#         self.biases[i] = self.biases[i] + dB[i]
      self.Wih += update[0]
      self.Whh += update[1]
      self.Who += update[2]
      self.bh += update[3]
      self.bo += update[4]


   def activate(self, sequence, initial_hidden = None):
      """
      Calculate the activation of each layer
      """

      if initial_hidden == None:
         hidden_activation = np.zeros((self.Whh.shape[1],1))
      else:
         hidden_activation = initial_hidden

      activations = []

      # Loop through the data.  The activations will consist of [[input_0, hidden_0, output_0], [input_1, hidden_1, output_1]...]
      for data in sequence:      
         hidden_activation = np.dot(self.Wih, data) + np.dot(self.Whh, hidden_activation) + self.bh
         output_activation = np.dot(self.Who, hidden_activation) + self.bo

         activations.append([data, hidden_activation, output_activation])
      # For this version, we'll assume sigmoid activation units
#      activations = [np.array( [data] ).transpose()]      # Makes it an Nx1 column vector

#      for i in range(1, self.numLayers):
         # The activation of each layer is simply the activation of the output of the
         # previous layer plus the bias
#         net = self.biases[i] + np.dot(self.weights[i], activations[-1])
#         activations.append(self.activation_functions[i](net))

      return activations
         


   def predict(self, sequence, initial_hidden = None):
      """
      Calculate the output layer for the given peice of data
      """

      activations = self.activate(sequence, initial_hidden)

      return [a[2] for a in activations]



