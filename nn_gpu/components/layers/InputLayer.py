## InputLayer.py	Dana Hughes		01-Sept-2015
##
## A simple layer which allows for providing an input to the network for the user.
## This layer contains only an output port, and doesn't really do much except convert
## from interfacing with the user to interfacing with the rest of the network
##
## History:
##	1.0	01-Sept-2015	Initial version.
##	1.01	03-Sept-2015	Adjusted to include Ports.

from AbstractLayer import AbstractLayer
from AbstractLayer import InputPort, OutputPort
import numpy as np
import gnumpy as gpu

class InputLayer(AbstractLayer):
   """
   An input layer
   """

   def __init__(self, inputSize):
      """
      Create an input layer, with batchSize rows and inputSize columns
      """

      # Properly inherit the AbstractLayer
      AbstractLayer.__init__(self)

      # This layer only has a output port. 
      self.output = OutputPort(inputSize)


   def setInput(self, inputBatch):
      """
      Set the input to the provided batch
      """

      # Set the output of the output port to the provided batch
      self.output.setOutput(inputBatch)


   def forward(self):
      """
      Perform a forward step
      """

      # Since there's only the output port, there's nothing to do
      pass


   def backward(self):
      """
      Perform a backprop step
      """

      # Input layers have no need to backprop error -- there is nothing 
      # to backprop to
      pass

