from nn.components.layers.RecurrentLayer import *
from nn.components.layers.LinearLayer import *
from nn.components.layers.InputLayer import *
from nn.components.connections.FullConnection import *
from nn.components.objectives.MSEObjective import *

import numpy as np

history = np.zeros((1,3))

il = InputLayer(1)
tl = InputLayer(1)

ll = LinearLayer(3)
rl = RecurrentLayer(ll, history)
ol = LinearLayer(1)

c1 = FullConnection(il.output, rl.input)
c2 = FullConnection(rl.output, ol.input)

c1.randomize()
c2.randomize()
rl.recurrentConnection.randomize()

obj = MSEObjective(ol.output, tl.output)

seq = np.array([[1],[1],[1],[2],[2],[2],[1],[1],[1],[2],[2],[2]])
tgt = np.array([[1],[1],[2],[2],[2],[1],[1],[1],[2],[2],[2],[1]])

def fwd():
   il.forward()
   tl.forward()
   c1.forward()
   rl.forward()
   c2.forward()
   ol.forward()
   obj.forward()

def bwd():
   obj.backward()
   ol.backward()
   c2.backward()
   rl.backward()
   c1.backward()
   tl.backward()
   il.backward()

def update():
   c1.updateParameterGradient()
   c2.updateParameterGradient()
   rl.recurrentConnection.updateParameterGradient()

def train_seq():
   # K, I'm tired, but this is how to do it.  For each RL, keep a history of the history activations.
   # Mostly works - doesn't predict next step... :(
   tot_obj = 0.0
   act_hist = []
   # Reset the rl
   rl.reset()

   c1.reset()
   c2.reset()
   rl.recurrentConnection.reset()

   for i in range(len(seq)):
      # Store the current history activation (from t-1)
      act_hist.append(rl.historyOutput.getOutput())
      # Set inputs and forward pass
      il.setInput(seq[i])
      tl.setInput(tgt[i])
      fwd()
      tot_obj += obj.getObjective()
      # Step forward with rl
      rl.step()

      # Can to objective, etc, too

   print "Objective - ", tot_obj

   # Now, to do backward, set the delta history to zero
   rl.setHistoryDelta(np.zeros((1,3)))
   # go backwards through the sequence
   for i in range(len(seq)-1, -1, -1):
     il.setInput(seq[i])
     tl.setInput(tgt[i])
     # Also set the history output to whatever it was at this time
     rl.historyOutput.setOutput(act_hist[i])
     # Perform forward and backward passes and update the gradient
     fwd()
     bwd()
     update()
     # And backstep to propagate recurrent delta backwards
     rl.backstep()

   # And update your gradients!
   c1.updateParameters(-0.1*c1.getParameterGradient() / 12)
   c2.updateParameters(-0.1*c2.getParameterGradient() / 12)
   rl.recurrentConnection.updateParameters(-0.1*rl.recurrentConnection.getParameterGradient() / 12)

