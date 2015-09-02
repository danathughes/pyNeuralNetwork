## Layer.py	Dana Hughes		01-Sept-2015
##
## An abstract interface for layers.  Every layer must implement the following
## methods:  forward, backward, reset, updateParameters, getParameters, getInput, getOutput,
##           getGradient, setInput(?), setOutput(?)

import numpy as np

class InputLayer:
   """
   An input layer
   """

   def __init__(self, batchSize, inputSize):
      """
      """

      pass


   def forward(self):
      """
      Perform a forward step
      """

      pass


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

      pass


   def getInput(self):
      """
      Provide the input to this unit
      """

      pass


   def getOutput(self):
      """
      Provide the output from this unit
      """

      pass


   def getGradient(self):
      """
      Return the gradient after backpropagation
      """

      pass

