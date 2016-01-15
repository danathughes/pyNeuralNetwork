## MSEObjective.py
##
## An objective layer which calculates the mean squared error between 
## predicted values and a target value.

import numpy as np
import gnumpy as gpu

class MSEObjective:
   """
   Objective layer which calculates the mean squared error between 
   predictions and targets
   """

   def __init__(self, output_port, target_port):
      """
      Create a new instance of an MSE objective.
      """

      self.objective = None
      self.delta = None

      # Connect to the ports and mirror the connection
      self.output_port = output_port
      self.target_port = target_port

      self.output_port.addConnection(self)
      self.target_port.addConnection(self)


   def setOutputPort(self, port):
      """
      Connect the objective to its output layer.
      """

      self.output_port = port
      self.output_port.addConnection(self)


   def setTargetPort(self, port):
      """
      Connect the objective to its target layer.
      """

      self.target_port = port
      self.target_port.addConnection(self)


   def getObjective(self):
      """
      Get the current objective value.
      """

      return self.objective


   def forward(self):
      """
      Perform a forward pass to calculate the activation (objective)
      """

      numExamples = self.output_port.getOutput().shape[0]
      self.objective = 0.5 * gpu.sum((self.output_port.getOutput() - self.target_port.getOutput())**2) / numExamples


   def backward(self):
      """
      Perform a backward pass to calculate the delta of this module
      """

      numExamples = self.output_port.getOutput().shape[0]
      self.delta = (self.output_port.getOutput() - self.target_port.getOutput()) / numExamples


   def getParameterGradient(self):
      """
      Return the gradient of the parameters for this module
      """

      # This objective function has no parameters
      return None


   def getDelta(self):
      """
      Return the delta after backward
      """

      return self.delta
