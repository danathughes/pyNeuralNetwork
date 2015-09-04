## AbsoluteValueLayer.py	Dana Hughes		03-Sept-2015
##
## A layer which implements absolute value activation function
##
## History:
##	1.0	03-Sept-2015	Initial version.


from AbstractLayer import AbstractLayer
from AbstractLayer import InputPort, OutputPort
import numpy as np

class AbsoluteValueLayer(AbstractLayer):
   """
   A layer which implements absolute value activation
   """

   def __init__(self, layerSize):
      """
      An absolute value layer can be connected to several inputs
      """

      # Properly inherit the AbstractLayer
      AbstractLayer.__init__(self)

      # A sigmoid layer has an input port and output port
      self.input = InputPort(layerSize)
      self.output = OutputPort(layerSize)


   def forward(self):
      """
      Perform a forward step - activate the net input using absolute value
      """

      # Perform the activation (logistic function)
      self.output.setOutput(np.abs(self.input.getNetInput()))


   def backward(self):
      """
      Perform a backprop step - gradient is the sign of the output
      """
            
      self.input.setDelta(np.sign(self.output.getOutput()) * self.output.getNetDelta())
