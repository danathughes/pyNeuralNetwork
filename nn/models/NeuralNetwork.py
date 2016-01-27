## NeuralNetwork.py    Dana Hughes    version 1.02     24-November-2014
##
## Fully connected neural network model, which predicts a output vector
## given an input vector.
##
## Revisions:
##   1.0   24-Nov-2014	Initial version, modified from LogisticRegressionModel with
##         		batch and stochastic gradient descent.
##   1.01  		Got backprop algorithm to work! 
##   1.02  		Added capability to have different activations
##   1.03  		Add weight decay 
##   1.04  		Pull out training functionality to work with Training.training
##   1.05  		Separate biases into a separate array, to make things a bit more 
##         		clear.  Fix back propagation, and make it all follow the
##         		logistic regression interface.
##   1.06  		Create functions for cost, activiation and activation gradients
##         		Abstracted activations, costs and gradients in neural network
##   1.1   03-Sept-2015	Changed class to interface with the new modules 
## 			(layers, connections, objectives, etc.)  Now the module
##			is simply a container for the various components.

import numpy as np
import random

class NeuralNetwork(object):
   """
   A container for feed forward neural networks
   """

   def __init__(self):
      """
      Create a new neural network container
      """

      # Maintain a list of the components
      self.layers = []
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


   def setParameters(self, parameters):
      """
      Assign the values of each of the parameters in the dictionary to the
      corresponding weights
      """

      for connection in parameters.keys():
         connection.setParameters(parameters[connection])


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


   def gradient(self, dataset, flatten=False):
      """
      Perform a training step on this model to get the gradient w.r.t. the dataset
      """

      # The training data should consist of inputs and targets
      inputs = dataset[0]
      targets = dataset[1]

      self.setInput(inputs)
      self.setTarget(targets)

      # Reset the gradients, perform forward and backward passes, and update
      self.reset()
      self.forward()
      self.backward()
      self.update()

      if flatten:
         return self.getFlatParameterGradients()
      else:
         return self.getParameterGradients()


   def getFlatParameters(self):
      """
      Provide a flattened (1-D) set of weights
      """

      # Construct a flattened version starting with an empty array
      parameters = np.array([])

      # Concatenate each flattened weight matrix to the parameters
      for connection in self.connections:
         parameters = np.concatenate([parameters, connection.getParameters().copy().flatten()])

      return parameters


   def setFlatParameters(self, parameters):
      """
      Set the parameters to the flattened (1-D) parameters provided
      """

      # Maintain the current index in the flattened parameters, and reconstruct
      # each parameter
      idx = 0

      for connection in self.connections:
         connection.setParameters(parameters[idx:idx+connection.getParameters().size].reshape(connection.getParameters().shape))
         idx += connection.getParameters().size


   def updateFlatParameters(self, update):
      """
      Update the parameters with a flattened update
      """

      idx = 0

      for connection in self.connections:
         connection.updateParameters(update[idx:idx+connection.getParameters().size].reshape(connection.getParameters().shape))
         idx += connection.getParameters().size

   def getFlatParameterGradients(self):
      """
      Get the current gradients of each connection
      """

      # Construct a flattened version starting with an empty array
      gradients = np.array([])

      # Concatenate each flattened weight matrix to the parameters
      for connection in self.connections:
         gradients = np.concatenate([gradients, connection.getParameterGradient().copy().flatten()])

      return gradients



