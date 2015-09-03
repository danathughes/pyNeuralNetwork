## AbstractConnection.py	Dana Hughes		02-Sept-2015
##
## An abstract interface for connections.  Connections all implement the following methods
## to be used with neural netwokrs:
##
##   forward()                 - Perform a forward pass (i.e., calculate the activation)
##   backward()                - Perform a backward pass (i.e., calculate the delta / gradient)
##   getInput()                - Get the current input to this layer
##   getOutput()               - Get the current output from this layer
##   getDelta()                - Get the current delta
##   setFromLayer()            - Connect the layer to the input of this connection
##   setToLayer()              - Connect the layer to the output of this connection
##   getParameters()           - Get the parameters which determine this module's behavior
##   updateParameters()        - Change the module's parameters by the provided amounts
##   reset()                   - Set module's parameter gradients to zero
##   getParameterGradient()    - Get the current parameter gradient
##   updateParameterGradient() - Calculate the parameter gradient after forward / backward pass
##
## Note that not all methods need be implemented.  For example, Biases do not have input,
## only output
##
## History:
##   1.00	02-Sept-2015	Initial version, mostly mimicing AbstractLayer
##

__author__ = 'Dana Hughes, dana.hughes@colorado.edu'


import numpy as np

class AbstractConnection(object):
   """
   An abstract connection, providing a common interface to all connections in a neural network
   """

   def __init__(self):
      """
      Create a Connection which contains no input, output, delta or gradient
      """

      # Initialize inputs, outputs, deltas and gradients to be None by default.
      # Then, getters and setters can easily be inherited
      self.input = None
      self.output = None

      self.delta = None

      self.gradient = np.zeros((0,0))
      self.parameters = np.zeros((0,0))

      # Connections typically have a from layer and a to layer
      self.from_layer = None
      self.to_layer = None


   def setFromLayer(self, layer):
      """
      Connect to the layer which feeds into this connection
      """

      self.from_layer = layer


   def setToLayer(self, layer):
      """
      Connect to the layer which this connection feeds into
      """

      self.to_layer = layer


   def forward(self):
      """
      Perform a forward step
      """

      pass


   def backward(self):
      """
      Perform a backprop step
      """

      pass


   def reset(self):
      """
      Set the gradient to zero
      """

      pass


   def updateParameters(self, dParams):
      """
      Update the parameters
      """

      pass


   def getParameters(self):
      """
      Return the weight matrix of this connection
      """

      return self.parameters


   def getInput(self):
      """
      Provide the input to this unit
      """

      return self.input()


   def getOutput(self):
      """
      Provide the output from this unit
      """

      return self.output


   def getParameterGradient(self):
      """
      Return the gradient after backpropagation
      """

      return self.gradient


   def updateParameterGradient(self):
      """
      Update the parameter gradient with the appropriate weight change based on forward and backward pass
      """

      pass


   def getDelta(self):
      """
      Return the delta after the backward pass
      """

      return self.delta
