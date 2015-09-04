## RecurrentLayer.py	Dana Hughes		01-Sept-2015
##
## A recurrent layer is a composite layer consisting of a layer and connection.  
##
## History:
##	1.0	03-Sept-2015	Initial version.


from AbstractLayer import AbstractLayer
from AbstractLayer import InputPort, OutputPort
from nn.components.connections.IdentityConnection import IdentityConnection
from nn.components.connections.FullConnection import FullConnection
import numpy as np


class MockConnectionOutput(object):
   """
   A dummy version of a connection output.  This allows for direct setting
   of a mock output connection, to interface with the base layer in a 
   recurrent layer
   """

   def __init__(self, port):
      """
      Make a new mock connection output and connect it to the port
      """

      # Start with a dummy value
      self.value = np.zeros((0,0))
      self.port = port
      self.port.addConnection(self)


   def setOutput(self, value):
      """
      Assign the output to the current value
      """

      self.value = value


   def getOutput(self):
      """
      Produce the current output
      """

      return self.value


class RecurrentLayer(AbstractLayer):
   """
   A layer which implements a delay in time
   """

   def __init__(self, baseLayer, initialHistory):
      """
      A recurrent layer extends the activation layer by adding a full recurrent
      connection from the output of the layer to its input, delayed by a 
      timestep.
      """

      # Properly inherit the AbstractLayer
      AbstractLayer.__init__(self)

      # Extract the layerSize from the provided activation layer
      self.baseLayer = baseLayer
      self.layerSize = self.baseLayer.input.size

      # A recurrent layer has an input port, history port and output port
      self.input = self.baseLayer.input
      self.history = InputPort(self.layerSize)
      self.output = self.baseLayer.output

      # Make two connections - the recurrent connection and a mock connection output from
      # the history to the activationLayer
      self.recurrentConnection = FullConnection(self.output, self.history)
      self.historyOutput = MockConnectionOutput(self.baseLayer.input)
      self.historyOutput.setOutput(initialHistory)

      # Keep track of how many timesteps there were, and the initial history incase of reset
      self.timestep = 0
      self.initialHistory = initialHistory


   def forward(self):
      """
      Perform a forward step - set the output to the current history
      """

      # Nothing much to do, simply call forward on the activation layer and recurrent connection
      self.baseLayer.forward()
      self.recurrentConnection.forward()


   def backward(self):
      """
      Perform the backprop step on the activation layer and recurrent connection
      """

      self.baseLayer.backward()
      self.recurrentConnection.backward()


   def step(self):
      """
      Step forward in time.  Propagate the history input to the history output
      """

      self.timestep += 1
      self.historyOutput.setOutput(self.history.getNetInput())


   def reset(self):
      """
      Set the history to the original initial history and set timestep to zero
      """

      self.timestep = 0
      self.historyOutput.setOutput(self.initialHistory)


   def getRecurrentConnection(self):
      """
      """
      return self.recurrentConnection


   def setHistoryDelta(self, delta):
      """
      """

      self.history.setDelta(delta)


   def backstep(self):
      """
      Step backward in time.  Propagate the input delta to the history
      """

      self.history.setDelta(self.input.getDelta())
