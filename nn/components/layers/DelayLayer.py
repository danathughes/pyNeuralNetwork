## DelayLayer.py	Dana Hughes		01-Sept-2015
##
## A layer which implements a delay of one timestep.  Used with recurrent
## neural networks
##
## History:
##	1.0	03-Sept-2015	Initial version.


from AbstractLayer import AbstractLayer
from AbstractLayer import InputPort, OutputPort
import numpy as np

class DelayLayer(AbstractLayer):
   """
   A layer which implements a delay in time
   """

   def __init__(self, layerSize, initialHistory):
      """
      A delay layer can be connected to several inputs
      """

      # Properly inherit the AbstractLayer
      AbstractLayer.__init__(self)

      # A delay layer has an input port and output port
      self.input = InputPort(layerSize)
      self.output = OutputPort(layerSize)

      # A delay layer has a history, which propagates forward
      # when step is called
      self.initial_history = initialHistory
      self.history = np.zeros((1,layerSize))
      self.current_step = 0


   def forward(self):
      """
      Perform a forward step - set the output to the current history
      """

      # Is this the first timestep?  Then adjust the shape of history to match
      # the shape of the input
      if self.current_step == 0:
         net_input = self.input.getNetInput()
         self.history = np.zeros(net_input.shape)
         self.history[:] = self.initial_history

      # Propagate the history forward, and set the input to the history
      self.output.setOutput(self.history)


   def backward(self):
      """
      Perform the backprop step - simply shift the delta backward
      """

      self.input.setDelta(self.output.getNetDelta())

   def step(self):
      """
      Step forward in time.  Set the history to the current input
      """

      self.history = self.input.getNetInput()
      self.current_step += 1
      
