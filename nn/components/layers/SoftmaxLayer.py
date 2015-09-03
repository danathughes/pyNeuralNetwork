## SoftmaxLayer.py		Dana Hughes		01-Sept-2015
##
## A layer which implements softmax function


from AbstractLayer import AbstractLayer
import numpy as np

class SoftmaxLayer(AbstractLayer):
   """
   A layer which implements sigmoid activation
   """

   def __init__(self):#, layerSize):
      """
      A sigmoid layer can be connected to several inputs
      """

      # Properly inherit the AbstractLayer
      AbstractLayer.__init__(self)


   def forward(self):
      """
      Perform a forward step - activate the net input using logistic function
      """

      # Calculate the net input to this layer
      self.input = sum([in_connection.getOutput() for in_connection in self.input_connections])

      # Perform the activation
      self.output = np.exp(self.input)
      self.output = self.output / (np.array([np.sum(self.output,1)]).transpose())


   def backward(self):
      """
      Perform a backprop step - gradient is the derivative of the sigmoid functon
      """
            
      self.delta = self.output * (1.0 - self.output) * sum([out.getDelta() for out in self.output_connections])
