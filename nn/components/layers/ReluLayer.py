## ReluLayer.py		Dana Hughes		02-Sept-2015
##
## A layer which implements a soft rectified linear activation function.
##
## History:
##	1.0	02-Sept-2015	Initial version.
##	1.01	03-Sept-2015	Adjusted to include Ports.


from AbstractLayer import AbstractLayer
import numpy as np

class ReluLayer(AbstractLayer):
   """
   A layer which implements rectified linear activation
   """

   def __init__(self, layerSize):
      """
      """

      # Properly inherit the AbstractLayer
      AbstractLayer.__init__(self)

      # A ReLU layer has an input port and output port
      self.input = InputPort(layerSize)
      self.output = OutputPort(layerSize)


   def forward(self):
      """
      Perform a forward step - activate the net input using the soft ReLU function
      """

      # Perform the activation (set any negative values to zero)
      self.output.setOutput(np.fmax(0.0, self.input.getNetInput()))


   def backward(self):
      """
      Perform a backprop step - gradient is simply 1 where the data is positive
      """
            
      self.input.setDelta(np.where(self.output.getOutput() > 0, 1.0, 0.0) * self.output.getNetDelta())
