## SoftReluLayer.py	Dana Hughes		02-Sept-2015
##
## A layer which implements a soft rectified linear activation function
##
## History:
##	1.0	02-Sept-2015	Initial version.
##	1.01	03-Sept-2015	Adjusted to include Ports.


from AbstractLayer import AbstractLayer
import numpy as np

class SoftReluLayer(AbstractLayer):
   """
   A layer which implements sigmoid activation
   """

   def __init__(self, layerSize):
      """
      """

      # Properly inherit the AbstractLayer
      AbstractLayer.__init__(self)

      # A sigmoid layer has an input port and output port
      self.input = InputPort(layerSize)
      self.output = OutputPort(layerSize)

   def forward(self):
      """
      Perform a forward step - activate the net input using the soft ReLU function
      """

      # Perform the activation (logistic function)
      self.output.setOutput(np.log(1.0 + np.exp(self.input.getNetInput())))


   def backward(self):
      """
      Perform a backprop step - gradient is the derivative of the sigmoid functon
      """
            
      self.input.setDelta((1.0 / (1.0 + np.exp(-self.output.getOutput()))) * self.output.getNetDelta())

