## SigmoidLayer.py	Dana Hughes		01-Sept-2015
##
## A layer which implements the sigmoid activation function.
##
## History:
##	1.0	01-Sept-2015	Initial version.
##	1.01	03-Sept-2015	Adjusted to include Ports.


from AbstractLayer import AbstractLayer
import numpy as np

class SigmoidLayer(AbstractLayer):
   """
   A layer which implements sigmoid activation
   """

   def __init__(self, layerSize):
      """
      A sigmoid layer can be connected to several inputs
      """

      # Properly inherit the AbstractLayer
      AbstractLayer.__init__(self)

      # A sigmoid layer has an input port and output port
      self.input = InputPort(layerSize)
      self.output = OutputPort(layerSize)


   def forward(self):
      """
      Perform a forward step - activate the net input using logistic function
      """

      # Perform the activation (logistic function)
      self.output.setOutput(1.0 / (1.0 + np.exp(-self.input.getNetInput())))


   def backward(self):
      """
      Perform a backprop step - gradient is the derivative of the sigmoid functon
      """
            
      self.input.setDelta(self.output.getOutput() * (1.0 - self.output.getOutput()) * self.output.getNetDelta())
