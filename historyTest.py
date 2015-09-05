from nn.components.layers.RecurrentLayer import *
from nn.components.layers.LinearLayer import *
from nn.components.layers.InputLayer import *
from nn.components.connections.FullConnection import *
from nn.components.objectives.MSEObjective import *

import numpy as np

inp = InputLayer(1)
hist = HistoryLayer(3, np.zeros((1,3)))
conn = FullConnection(inp.output, hist.input)
tgt = InputLayer(3)
obj = MSEObjective(hist.output, tgt.output)

conn.parameters[0,0] = -0.3
conn.parameters[0,1] = 0.5
conn.parameters[0,2] = -0.2

inp.setInput(np.array([[1]]))

def print_state():
   print "  inp.output: ", inp.output.value
   print "  conn.output: ", conn.getOutput()
   print "  hist.output: ", hist.output.value
   print "  obj.delta: ", obj.getDelta()
   print "  hist.delta: ", hist.input.getDelta()
   print "  Current History: "
   for i in range(len(hist.history)):
      print "    ", hist.history[i]
   print

