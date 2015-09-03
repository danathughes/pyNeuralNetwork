## AbstractLayer.py	Dana Hughes		01-Sept-2015
##
## An abstract interface for layers.  Layers all implement the following methods
## to be used with neural netwokrs:
##
##   forward()                 - Perform a forward pass (i.e., calculate the activation)
##   backward()                - Perform a backward pass (i.e., calculate the delta / gradient)
##   getInput()                - Get the current input to this layer
##   getOutput()               - Get the current output from this layer
##   getDelta()                - Get the current delta
##   addInputConnection()      - Connect to the input of this layer
##   addOutputConneciton()     - Connect to the output of this layer
##   getParameters()           - Get the parameters which determine this module's behavior
##   updateParameters()        - Change the module's parameters by the provided amounts
##   reset()                   - Set module's parameter gradients to zero
##   getParameterGradient()    - Get the current parameter gradient
##   updateParameterGradient() - Calculate the parameter gradient after forward / backward pass
##
## Note that not all methods need be implemented.  Primarily, not all layers necessarily have any
## parameters, so anything associated with getting or updating parameters does nothing or returns
## nothing.


import numpy as np

class AbstractLayer:
   """
   An abstract input layer, providing a common interface to all layers in a neural network
   """

   def __init__(self):
      """
      Create a Layer which contains no input, output, delta or gradient
      """

      # Initialize inputs, outputs, deltas and gradients to be None by default.
      # Then, getters and setters can easily be inherited
      self.input = None
      self.output = None
      self.delta = None
      self.gradient = None
      self.parameters = None

      # Layers have input and output connections.  Since there can be multiple connections,
      # Maintain a list of input and output connections
      self.input_connections = []
      self.output_connections = []


   def addInputConnection(self, connection):
      """
      Add a connection to the input of this layer
      """

      self.input_connections.append(connection)


   def addOutputConnection(self, connection):
      """
      Add a connection to the output of this layer
      """

      self.output_connections.append(connection)


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


   def updateParameters(self, params):
      """
      Update the parameters
      """

      pass


   def getParameters(self):
      """
      Return the gradient of this layer
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
