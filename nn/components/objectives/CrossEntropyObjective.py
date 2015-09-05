## CrossEntropyObjective.py
##
## An objective layer which calculates the cross entropy between predicted 
## values and a target value.  Usually used in conjunction with softmax
## layers for classification purposes

import numpy as np

class CrossEntropyObjective:
   """
   Objective layer which calculates the cross entropy error between 
   predictions and targets
   """

   def __init__(self, output_port, target_port):
      """
      Create a new instance of a cross entropy objective.
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
      self.objective = -np.sum(self.target_port.getOutput() * np.log(self.output_port.getOutput()))
      self.objective += -np.sum((1.0 - self.target_port.getOutput())*(np.log(1.000001 - self.output_port.getOutput())))
      self.objective /= numExamples

   def backward(self):
      """
      Perform a backward pass to calculate the delta of this module
      """

      numExamples = self.output_port.getOutput().shape[0]
      self.delta = (self.output_port.getOutput() - self.target_port.getOutput())
#      self.delta /= (self.output_port.getOutput() * (1.0 - self.output_port.getOutput()))
      self.delta /= numExamples


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
