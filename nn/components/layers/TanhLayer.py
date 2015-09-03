## TanhLayer.py		Dana Hughes		01-Sept-2015
##
## A layer which implements hyperbolic activation function


from AbstractLayer import AbstractLayer
import numpy as np

class TanhLayer(AbstractLayer):
   """
   A layer which implements sigmoid activation
   """

   def __init__(self):#, layerSize):
      """
      A sigmoid layer can be connected to several inputs
      """

      # Properly inherit the AbstractLayer
      AbstractLayer.__init__(self)

#      self.input = np.zeros((1, layerSize))
#      self.output = np.zeros((1, layerSize))
#      self.delta = np.zeros((1, layerSize))


   def forward(self):
      """
      Perform a forward step - activate the net input using logistic function
      """

      # Calculate the net input to this layer
      self.input = sum([in_connection.getOutput() for in_connection in self.input_connections])

      # Perform the activation (logistic function)
      self.output = (1.0 - np.exp(-self.input)) / (1.0 + np.exp(-self.input))


   def backward(self):
      """
      Perform a backprop step - gradient is the derivative of the sigmoid functon
      """
            
      self.delta = (1.0 - self.output**2) * sum([out.getDelta() for out in self.output_connections])
