## RecurrentNeuralNetwork.py    Dana Hughes    version 1.0     04-Sept-2015
##
## Recurrent neural network model.
##
## Revisions:
##   1.0   04-Sept-2015	Initial version, modified from NeuralNetwork model.

import numpy as np
import random

class RecurrentNeuralNetwork(object):
   """
   A container for feed forward neural networks
   """

   def __init__(self):
      """
      Create a new recurrent neural network container
      """

      # Maintain a list of the components
      self.layers = []
      self.recurrentlayers = []
      self.connections = []
      self.objectives = []

      # What is the order of the modules
      self.module_order = []

      # Keep a special pointer to the input, output and objective
      self.input_layer = None
      self.output_layer = None
      self.objective_layer = None
     

   def addLayer(self, layer):
      """
      Add a layer to the model
      """

      self.layers.append(layer)
      self.module_order.append(layer)


   def addRecurrentLayer(self, layer):
      """
      Add a recurrent layer to the model
      """

      self.recurrentlayers.append(layer)

      # The user may or may not have added this to the layers as well,
      # which is necessary as that list gets called for other things
      if not layer in self.layers:
         self.layers.append(layer)

      self.module_order.append(layer)

      # Also, pull out the recurrentConnection and add to the connections
      self.connections.append(layer.getRecurrentConnection())
      


   def addConnection(self, connection):
      """
      Add a connection to the model
      """

      self.connections.append(connection)
      self.module_order.append(connection)


   def addObjective(self, objective):
      """
      Add an objective to the model
      """
  
      self.objectives.append(objective)
      self.module_order.append(objective)


   def setInputLayer(self, layer):
      """
      Identify which layer is the model's input
      """

      self.input_layer = layer


   def setTargetLayer(self, layer):
      """
      Identify which layer is the model's target for training
      """

      self.target_layer = layer


   def setOutputLayer(self, layer):
      """
      Identify which layer is the model's output
      """

      self.output_layer = layer


   def setObjective(self, objective):
      """
      Identify which layer is the model's overall objective
      """

      self.objective_layer = objective


   def randomize(self):
      """
      Initialize the parameters of connections to small random values
      """

      for connection in self.connections:
         connection.randomize()


   def prepare(self):
      """
      Get the neural network ready for training
      """

      # We'll get to this, it'll basically determine the module order using
      # some sort of graph numbering algorithm

      pass


   def reset(self):
      """
      Reset the gradient of every connection to prepare for a new training cycle
      """

      for connection in self.connections:
         connection.reset()

      # Recurrent layers also need to be reset to clear their histories
      for layer in self.recurrentlayers:
         layer.reset()
     
 

   def setInput(self, inputBatch):
      """
      Set the input to the new values
      """

      self.input_layer.setInput(inputBatch)


   def setTarget(self, targetBatch):
      """
      Set the target to the new values
      """

      self.target_layer.setInput(targetBatch)


   def getOutput(self):
      """
      Get the current output
      """

      return self.output_layer.output.getOutput()


   def getObjective(self):
      """
      Get the current objective value
      """

      return self.objective_layer.getObjective()


   def getSequenceObjective(self, sequence):
      """
      Compute the objective for the entire sequence
      """

      input_sequence = sequence[0]
      target_sequence = sequence[1]

      # Sequence length and batch size
      sequence_length = input_sequence.shape[0]
      batch_size = input_sequence.shape[1]

      # Reset everything to do forward passes and getting objectives
      self.reset()

      # Set the initial history to zero 
      for layer in self.recurrentlayers:
         layer.zeroInitialHistoryBatch(batch_size)      

      total_objective = 0.0
      
      for t in range(sequence_length):
         # Do a forward pass and add the objective at this layer.
         self.setInput(input_sequence[t,:])
         self.setTarget(target_sequence[t,:])
         self.forward()
         self.step()
         total_objective += self.getObjective()

      return total_objective


   def forward(self):
      """
      Perform a forward pass 
      """

      for module in self.module_order:
         module.forward()


   def backward(self):
      """
      Perform a backward pass
      """

      # Reverse the module_order, perform backwards, then reverse again
      self.module_order.reverse()

      for module in self.module_order:
         module.backward()

      self.module_order.reverse()


   def step(self):
      """
      Perform a time step on the recurrent layers
      """

      for layer in self.recurrentlayers:
         layer.step()


   def backstep(self):
      """
      Perform a time backstep on the recurrent layers
      """

      for layer in self.recurrentlayers:
         layer.backstep()


   def update(self):
      """
      Perform an update on all parameter gradients
      """

      for connection in self.connections:
         connection.updateParameterGradient()


   def getParameters(self):
      """
      Provide the parameters (weights) of each 
      """

      parameters = {}

      for connection in self.connections:
         parameters[connection] = connection.getParameters()

      return parameters


   def updateParameters(self, updates):
      """
      Update the weights in the model by adding dW
      """

      for connection, update in updates.items():
         connection.updateParameters(update)


   def getParameterGradients(self):
      """
      Get the current gradients of each connection
      """

      gradients = {}

      for connection in self.connections:
         gradients[connection] = connection.getParameterGradient()

      return gradients


   def gradient(self, dataset, normalize = False):
      """
      Perform a training step on this model to get the gradient w.r.t. the dataset
      """

      # The training data should consist of input and target sequences.  These are tensors
      # with dimension seq_length x batch_size x vector_size
      input_sequence = dataset[0]
      target_sequence = dataset[1]

      batch_size = input_sequence.shape[1]
      seq_length = input_sequence.shape[0]

      # Set the initial history to zero 
      for layer in self.recurrentlayers:
         layer.zeroInitialHistoryBatch(batch_size)

      # Reset the network
      self.reset()

      # Perform forward pass on the sequence
      for t in range(seq_length):
         # Set the input and target to the current timestep
         self.setInput(input_sequence[t,:])
         self.setTarget(target_sequence[t,:])

         # Perform a forward pass and step
         self.forward()
         self.step()

      # Set the initial history delta to zero
      for layer in self.recurrentlayers:
         layer.zeroHistoryDeltaBatch(batch_size)

      # Now go backwards in time computing deltas
      for t in range(seq_length-1, -1, -1):
         # Go backwards in time propagating the deltas backwards
         self.backstep()

         # Perform a forward and backward pass, using the input and
         # target at this time step, then update parameters
         self.setInput(input_sequence[t,:])
         self.setTarget(target_sequence[t,:])
         self.forward()
         self.backward()
         self.update()
         
      # All done, return the gradients, scaled down by the sequence length if
      # normalization is desired
      gradients = self.getParameterGradients()

      if normalize:
         for units in gradients.keys():
            gradients[units] /= seq_length

      return gradients
