## SplitLayer.py	Dana Hughes		08-Sept-2015
##
## A layer which is used to split a single input to multiple outputs
##
## History:
##	1.0	08-Sept-2015	Initial version.


from AbstractLayer import AbstractLayer
from AbstractLayer import InputPort, OutputPort
import numpy as np

class SplitLayer(AbstractLayer):
   """
   A layer which splits an input port into multiple output ports
   """

   def __init__(self, inputSize, outputSizes):
      """
      Create a layer which splits the input into the provided output sizes
      """

      # Properly inherit the AbstractLayer
      AbstractLayer.__init__(self)

      # Should probably have an assertion that the output sizes add up to the
      # input sizes

      # A sigmoid layer has an input port and output port
      self.input = InputPort(inputSize)
      self.outputPorts = []
      for size in outputSizes:
         self.outputPorts.append(OutputPort(size))


   def forward(self):
      """
      Perform a forward step - split the input to the various outputs
      """

      # We'll iterate through the ports, splitting the input among them
      idx = 0

      for port in self.outputPorts:
         port.setOutput(self.input.getNetInput()[:,idx:idx+port.size])
         idx += port.size


   def backward(self):
      """
      Perform a backprop step - join the net deltas together to get the input delta
      """

      # We'll iterate through the output ports, getting each delta
      deltas = np.zeros(self.input.value.shape)
      idx = 0

      for port in self.outputPorts:
         deltas[:,idx:idx+port.size] = port.getNetDelta()
            
      self.input.setDelta(deltas)

