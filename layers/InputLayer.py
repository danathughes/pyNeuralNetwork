## InputLayer.py	Dana Hughes		01-Sept-2015
##
##

import numpy as np

class InputLayer:
   """
   An input layer
   """

   def __init__(self, batchSize, inputSize):
      """
      Create an input layer, with batchSize rows and inputSize columns
      """

      # Initialize the input (user provided) and output (after forward pass)
      # Note that output is simply equivalent to the input
      self.inputs = np.zeros((batchSize, inputSize))
      self.output = self.inputs

      # There is no gradient to keep track of
      self.gradient = np.zeros((0,0))

      self.shape = (batchSize, inputSize)

      self.input_connections = []
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

      pass


   def reset(self):
      """
      Set the gradient to zero
      """

      pass


   def updateParameters(self, params):
      """
      Update the parameters
      """

      pass


   def getParameters(self):
      """
      Return the gradient of this layer
      """

      return self.gradient


   def getInput(self):
      """
      Provide the input to this unit
      """

      return self.input


   def getOutput(self):
      """
      Provide the output from this unit
      """

      return self.output


   def getGradient(self):
      """
      This is input, there is not gradient
      """

      return np.zeros(self.shape)

