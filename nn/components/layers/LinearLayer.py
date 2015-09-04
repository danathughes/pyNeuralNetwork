## LinearLayer.py	Dana Hughes		01-Sept-2015
##
## A layer which has not activation function, simply stores values
##
## History:
##	1.0	03-Sept-2015	Initial version.


from AbstractLayer import AbstractLayer
from AbstractLayer import InputPort, OutputPort
import numpy as np

class LinearLayer(AbstractLayer):
   """
   A linear layer which has not activation function
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
      self.output.setOutput(self.input.getNetInput())


   def backward(self):
      """
      Perform a backprop step - gradient is the derivative of the sigmoid functon
      """
            
      self.input.setDelta(self.output.getNetDelta())

