## InputLayer.py	Dana Hughes		01-Sept-2015
##
##

from AbstractLayer import AbstractLayer
import numpy as np

class InputLayer(AbstractLayer):
   """
   An input layer
   """

   def __init__(self, batchSize, inputSize):
      """
      Create an input layer, with batchSize rows and inputSize columns
      """

      # Properly inherit the AbstractLayer
      AbstractLayer.__init__(self)

      # Initialize the input (user provided) and output (after forward pass)
      # Note that output is simply equivalent to the input
      self.inputs = np.zeros((batchSize, inputSize))
      self.output = self.inputs

      self.shape = (batchSize, inputSize)

      self.output_connections = []


   def setInput(self, batch):
      """
      Set the input to the provided batch
      """

      # Check that the provided data actually fits into what we have set up
#      assert inputs.shape = self.shape, "Batch shape (%d, %d) does not equal input layer shape (%d, %d)" % (batch.shape[0], batch.shape[1], self.shape[0], self.shape[1])

      self.inputs[:] = batch


   def forward(self):
      """
      Perform a forward step
      """

      self.output = self.inputs


   def backward(self):
      """
      Perform a backprop step
      """

      # Input layers have no need to backprop error -- there is nothing 
      # to backprop to
      pass

