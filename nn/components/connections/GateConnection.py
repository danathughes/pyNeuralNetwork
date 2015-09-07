## GateConnection.py		Dana Hughes		07-Sept-2015
##
## A gated identity connection, which basically implements a multiplicative
## unit.  The gate neuron is assumed to be connected to a Sigmoid unit, so
## that multiplication is between 0 (off) and 1 (on).
##
## History:
##   1.0	07-Sept-2015	Initial Version

import numpy as np
from AbstractConnection import AbstractConnection

class GateConnection(AbstractConnection):
   """
   A gated identity connection.
   """

   def __init__(self, from_port, to_port, gate_port):
      """
      Create a new identity connection
      """

      # Properly initialize the abstract connection
      AbstractConnection.__init__(self, from_port, to_port)

      # Determine the dimensions and initialize the weight matrix
      self.dimensions = (from_port.size, to_port.size)

      self.gate_port = gate_port

      self.input = np.zeros((1,from_port.size))
      self.output = np.zeros((1,to_port.size))
      self.gate = np.zeros((1,1))
      self.delta = np.zeros((1,from_port.size))
      self.gate_delta = np.zeros((1,1))

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
      self.output = self.input * self.gate_port.getOutput()


   def backward(self):
      """
      Perform a backprop step - directly propagate deltas backward
      """

      self.delta = self.to_port.getDelta() * self.gate_port.getOutput()
      self.gate_delta = np.sum(self.to_port.getDelta() * self.from_port.getOutput()) / self.dimensions[0]


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

