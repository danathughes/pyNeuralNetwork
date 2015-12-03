## RecurrentLayer.py	Dana Hughes		01-Sept-2015
##
## A recurrent layer is a composite layer consisting of a layer and connection.  
##
## History:
##	1.0	03-Sept-2015	Initial version.
##	1.01	04-Sept-2015	Modified internal workings, replacing a MockConnection
##				with a HistoryLayer class.  Also, allowed arbitrary
##				Layer and Connection classes to be used.


from AbstractLayer import AbstractLayer
from AbstractLayer import InputPort, OutputPort
from nn_gpu.components.connections.IdentityConnection import IdentityConnection
from nn_gpu.components.connections.FullConnection import FullConnection
from nn_gpu.components.layers.SigmoidLayer import SigmoidLayer
import numpy as np
import gnumpy as gpu

## HistoryLayer
##
##

class HistoryLayer(AbstractLayer):
   """
   A useful internal layer for Recurrent Layers which maintains a history
   of activations.
   """

   def __init__(self, size, initialHistory=gpu.zeros((0,0))):
      """
      Create a History layer
      """

      AbstractLayer.__init__(self)
      self.layerSize = size

      self.input = InputPort(self.layerSize)
      self.output = OutputPort(self.layerSize)

      self.history = []

      self.output.value = gpu.garray(np.copy(initialHistory.as_numpy_array()))
      self.initialHistory = initialHistory


   def forward(self):
      """
      Do nothing.  step handles this layer correctly
      """
  
      pass


   def backward(self):
      """
      Do nothing.  backstep handles this layer correctly
      """

      pass


   def step(self):
      """
      Push the current output into the history, and propagate input forward
      """

      self.history.append(self.output.value[:])
      self.output.value = self.input.getNetInput()


   def backstep(self):
      """
      Pop the output from the history, and propagate the delta backward
      """

      self.input.setDelta(self.output.getNetDelta())
      self.output.value = self.history.pop()


   def reset(self):
      """
      Reset the history to empty and output to initialHistory
      """

      self.history = []
      self.output.value[:] = self.initialHistory


   def setDelta(self, delta):
      """
      Set the delta on the input layer to the provided value
      """

      self.input.setDelta(delta)


## RecurrentLayer
##
##

class RecurrentLayer(AbstractLayer):
   """
   A layer which implements a delay in time
   """

   def __init__(self, size, initialHistory = gpu.zeros((0,0)), baseLayerClass = SigmoidLayer, connectionClass = FullConnection):
      """
      A recurrent layer extends the activation layer by adding a full recurrent
      connection from the output of the layer to its input, delayed by a 
      timestep.
      """

      # Properly inherit the AbstractLayer
      AbstractLayer.__init__(self)

      # Extract the layerSize from the provided activation layer
      self.baseLayer = baseLayerClass(size)
      self.layerSize = size
      self.historyLayer = HistoryLayer(self.layerSize, initialHistory)

      # A recurrent layer has an input port, history port and output port
      self.input = self.baseLayer.input
      self.output = self.baseLayer.output

      # Make two connections - the recurrent connection to the history and a connection from
      # the history to the activationLayer
      self.recurrentConnection = connectionClass(self.output, self.historyLayer.input)
      self.historyConnection = IdentityConnection(self.historyLayer.output, self.input)

      # Keep track of how many timesteps there were, and the initial history incase of reset
      self.timestep = 0


   def forward(self):
      """
      Perform a forward step - set the output to the current history
      """

      # Nothing much to do, simply call forward on the activation layer and recurrent connection
      self.historyLayer.forward()
      self.historyConnection.forward()
      self.baseLayer.forward()
      self.recurrentConnection.forward()


   def backward(self):
      """
      Perform the backprop step on the activation layer and recurrent connection
      """
      
      self.recurrentConnection.backward()
      self.baseLayer.backward()
      self.historyConnection.backward()
      self.historyLayer.backward()


   def step(self):
      """
      Step forward in time.  Propagate the history input to the history output
      """

      self.timestep += 1
      self.historyLayer.step()


   def reset(self):
      """
      Set the history to the original initial history and set timestep to zero
      """

      self.timestep = 0
      self.historyLayer.reset()
     

   def getRecurrentConnection(self):
      """
      Provide the recurrent connection
      """
      return self.recurrentConnection


   def setInitialHistory(self, history):
      """
      Setup the initial history of this layer
      """

      self.historyLayer.initialHistory = history
      self.historyLayer.output.value = gpu.garray(np.copy(history.as_numpy_array()))


   def setHistoryDelta(self, delta):
      """
      Set the delta on the history layer to the provided value
      """

      self.historyLayer.setDelta(delta)
      # We need to propagate the delta backward to the appropriate ports
      self.backward()


   def zeroInitialHistoryBatch(self, batchSize):
      """
      Set the initial history to zeros for the provided batch size
      """

      zero_history = gpu.zeros((batchSize, self.layerSize))
      self.setInitialHistory(zero_history)


   def zeroHistoryDeltaBatch(self, batchSize):
      """
      Set the initial history delta to zeros for the provided batch size
      """

      zero_delta = gpu.zeros((batchSize, self.layerSize))
      self.setHistoryDelta(zero_delta)


   def backstep(self):
      """
      Step backward in time.  Propagate the input delta to the history
      """

      self.timestep -= 1
      self.historyLayer.backstep()
