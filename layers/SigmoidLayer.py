## SigmoidLayer.py	Dana Hughes		01-Sept-2015
##
## A layer which implements the sigmoid activation function


import numpy as np

class SigmoidLayer:
   """
   A layer which implements sigmoid activation
   """

   def __init__(self, batchSize, inputSize):
      """
      A sigmoid layer can be connected to several inputs
      """

      self.shape = (batchSize, inputSize)

      self.input = np.zeros(self.shape)
      self.output = np.zeros(self.shape)
      self.gradient = np.zeros(self.shape)

      # What are the connections to this layer
      self.input_connections = []
      self.output_connections = []


   def forward(self):
      """
      Perform a forward step - activate the net input using logistic function
      """

      # Calculate the net input
      self.input *= 0.0
      for in_connection in self.input_connections:
         self.input += in_connection.getOutput()

      # Perform the activation (logistic function)
      self.output = 1.0 / (1.0 + np.exp(-self.input))


   def backward(self):
      """
      Perform a backprop step - gradient is the derivative of the sigmoid functon
      """

      self.gradient = self.output * (1.0 - self.output)


   def reset(self):
      """
      Set the gradient to zero
      """

      self.gradient = np.zeros(self.shape)


   def updateParameters(self, params):
      """
      Update the parameters
      """

      pass


   def getParameters(self):
      """
      Return the gradient of this layer
      """

      return np.zeros((0,0))


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
      Return the gradient after backpropagation
      """

      return self.gradient

