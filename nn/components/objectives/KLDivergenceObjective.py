## KLDivervenceObjective.py	Dana Hughes    version 1.0     07-Sept-2015
##
## An objective layer which calculates the KL divergence of its output port.  The KL 
## divergence measures an average activation for each neuron, and can be used to 
## provide sparsity for a particular layer.
##
## History:
##	07-Sept-2015	1.0	Initial Version

import numpy as np

class KLDivergenceObjective:
   """
   Objective layer which calculates the KL divergence of a layer.
   """

   def __init__(self, output_port, sparsity = 0.1, inactive_value = 0.0):
      """
      Create a new instance of a KL divergence objective.
      """

      self.objective = None
      self.delta = None

      # Connect to the ports and mirror the connection
      self.output_port = output_port

      self.output_port.addConnection(self)

      # Hang on to the sparsity parameter
      self.sparsity = sparsity
      self.inactive_value = inactive_value


   def setOutputPort(self, port):
      """
      Connect the objective to its output layer.
      """

      self.output_port = port
      self.output_port.addConnection(self)


   def getObjective(self):
      """
      Get the current objective value.
      """

      return self.objective


   def forward(self):
      """
      Perform a forward pass to calculate the activation (objective)
      """

      kl = np.sum(self.output_port.getOutput() - self.inactive_value, 0)
      kl /= self.output_port.getOutput().shape[0]
      self.objective = kl * np.log(kl/self.sparsity)
      self.objective += (1.0 - kl) * np.log((1.0 - kl) / (1.0 - self.sparsity))


   def backward(self):
      """
      Perform a backward pass to calculate the delta of this module
      """

      kl = np.sum(self.output_port.getOutput() - self.inactive_value, 0)
      kl /= self.output_port.getOutput().shape[0]
      self.delta = -kl/self.sparsity
      self.delta += (1.0 - kl) / (1.0 - self.sparsity)


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
