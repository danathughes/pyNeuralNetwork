## IdentityConnection.py	Dana Hughes		03-Sept-2015
##
## A simple identity connection which performs no transformation from
## input to output.
##
## History:
##   1.0	03-Sept-2015	Initial Version

import numpy as np
from AbstractConnection import AbstractConnection

class IdentityConnection(AbstractConnection):
   """
   A connection which fully links two layers.
   """

   def __init__(self, from_port, to_port):
      """
      Create a new identity connection
      """

      # Properly initialize the abstract connection
      AbstractConnection.__init__(self, from_port, to_port)

      # Determine the dimensions and initialize the weight matrix
      self.dimensions = (from_port.size, to_port.size)

      self.input = np.zeros((1,from_port.size))
      self.output = np.zeros((1,to_port.size))
      self.delta = np.zeros((1,from_port.size))

      # Dummy values to allow this to follow the AbstractConnection interface
      self.parameters = np.zeros((0,0))
      self.gradient = np.zeros((0,0))


   def randomize(self):
      """
      Does nothing - simply to follow AbstractConnection 
      """

      pass


   def forward(self):
      """
      Perform a forward step - simply map the input to the output
      """

      self.input = self.from_port.getOutput()
      self.output = self.input


   def backward(self):
      """
      Perform a backprop step - directly propagate deltas backward
      """

      self.delta = self.to_port.getDelta()


   def reset(self):
      """
      Does nothing
      """

      pass


   def updateParameters(self, dParams):
      """
      Does nothing
      """

      pass


   def updateParameterGradient(self):
      """
      Does nothing
      """

      pass

