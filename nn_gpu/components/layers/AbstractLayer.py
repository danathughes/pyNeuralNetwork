## AbstractLayer.py	Dana Hughes		01-Sept-2015
##
## An abstract interface for layers.  Layers all implement the following methods
## to be used with neural netwokrs:
##
##   forward()                 - Perform a forward pass (i.e., calculate the activation)
##   backward()                - Perform a backward pass (i.e., calculate the delta / gradient)
##   getParameters()           - Get the parameters which determine this module's behavior
##   updateParameters()        - Change the module's parameters by the provided amounts
##   reset()                   - Set module's parameter gradients to zero
##   getParameterGradient()    - Get the current parameter gradient
##   updateParameterGradient() - Calculate the parameter gradient after forward / backward pass
##
## Note that not all methods need be implemented.  Primarily, not all layers necessarily have any
## parameters, so anything associated with getting or updating parameters does nothing or returns
## nothing.
##
## History:
##   1.00	01-Sept-2015	Initial version, extracting common methods from all layers
##   1.01	03-Sept-2015	Introduce the concept of Ports, which basically act as interfaces
##				to the actual layer.  These are what connections will connect to.

__author__ = 'Dana Hughes, dana.hughes@colorado.edu'


import numpy as np
import gnumpy as gpu

## A Port is an input / output interface to a layer, and is what is connected to a Connection.
## Ports need to be able to update a value (i.e., the input or output) during a foward pass and
## update a delta during a backward pass.  Ports need to implement the following interface
##
##   addConnection()           - Ad a connection to this port


## 

class InputPort(object):
   """
   An input to a layer
   """

   def __init__(self, portSize):
      """
      Make a new input port to a layer
      """

      self.size = portSize
      self.delta = np.zeros((1, portSize))

      self.connections = []


   def addConnection(self, connection):
      """
      Add a connection to this input.  Connections must implement a getOutput method
      """

      self.connections.append(connection)


   def getNetInput(self):
      """
      Get the net input to this port by adding the output of all connections
      """

      return sum([gpu.garray(conn.getOutput()) for conn in self.connections])


   def setDelta(self, delta):
      """
      Set the delta of this port
      """

      self.delta = delta


   def getDelta(self):
      """
      Get the delta from this port.  Used during backward pass.
      """

      return self.delta


class OutputPort(object):
   """
   An output to a layer
   """

   def __init__(self, portSize):
      """
      Make a new output port for a layer
      """
 
      self.size = portSize
      self.value = np.zeros((1, portSize))

      self.connections = []


   def addConnection(self, connection):
      """
      Add a connection
      """

      self.connections.append(connection)


   def setOutput(self, value):
      """
      Set the output of this port to the new value
      """

      self.value = value


   def getOutput(self):
      """
      Get the current output at this port
      """

      return self.value


   def getNetDelta(self):
      """
      Calculate the net delta from connections to this port
      """

      return sum([conn.getDelta() for conn in self.connections])


class AbstractLayer(object):
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
      self.gradient = gpu.zeros((0,0))
      self.parameters = gpu.zeros((0,0))


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

