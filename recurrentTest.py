from nn.components.layers.RecurrentLayer import *
from nn.components.layers.SigmoidLayer import *
from nn.components.layers.InputLayer import *
from nn.components.connections.FullConnection import *
from nn.components.connections.Bias import *
from nn.components.objectives.MSEObjective import *

import numpy as np


init_hist = np.zeros((1,5))
inp = InputLayer(1)
rec = RecurrentLayer(5, init_hist)
out = SigmoidLayer(1)

i2r = FullConnection(inp.output, rec.input)
r2o = FullConnection(rec.output, out.input)
rbias = Bias(rec.input)
obias = Bias(out.input)

tgt = InputLayer(1)
obj = MSEObjective(out.output, tgt.output)


i2r.randomize()
r2o.randomize()
rbias.randomize()
obias.randomize()
rec.recurrentConnection.randomize()

def fwd():
   inp.forward()
   tgt.forward()
   i2r.forward()
   rbias.forward()
   rec.forward()
   r2o.forward()
   obias.forward()
   out.forward()
   obj.forward()


def bwd():
   obj.backward()
   out.backward()
   obias.backward()
   r2o.backward()
   rec.backward()
   rbias.backward()
   i2r.backward()
   tgt.backward()
   inp.backward()


def update():
   i2r.updateParameterGradient()
   rbias.updateParameterGradient()
   r2o.updateParameterGradient()
   obias.updateParameterGradient()
   rec.recurrentConnection.updateParameterGradient()


def grad_dec():
   g = i2r.getParameterGradient()
   i2r.updateParameters(-0.9*g)
   g = r2o.getParameterGradient()
   r2o.updateParameters(-0.9*g)
   g = rbias.getParameterGradient()
   rbias.updateParameters(-0.9*g)
   g = obias.getParameterGradient()
   obias.updateParameters(-0.9*g)
   g = rec.recurrentConnection.getParameterGradient()
   rec.recurrentConnection.updateParameters(-0.9*g)
   

def reset():
   rec.reset()
   i2r.reset()
   r2o.reset()
   rbias.reset()
   obias.reset()
   rec.recurrentConnection.reset()

seq = np.array([[[1]],[[1]],[[1]],[[0]],[[0]],[[0]],[[1]],[[1]],[[1]],[[0]],[[0]],[[0]]])
tar = np.array([[[1]],[[0]],[[1]],[[1]],[[1]],[[1]],[[0]],[[1]],[[0]],[[0]],[[0]],[[0]]])

def train_seq():
   tot_obj = 0.0
   reset()
   for i in range(12):
      inp.setInput(seq[i,:])
      tgt.setInput(tar[i,:])
      fwd()
      tot_obj += obj.getObjective()
      rec.step()
   print "Obj = ", tot_obj
   rec.setHistoryDelta(np.zeros((1,5)))
   for i in range(11,-1,-1):

      rec.backstep()
      inp.setInput(seq[i,:])
      tgt.setInput(tar[i,:])
      fwd()
      bwd()
      update()
   grad_dec()


def check():
   reset()
   for i in range(12):
      inp.setInput(seq[i,:])
      tgt.setInput(tar[i,:])
      fwd()
      opt = out.output.getOutput()
      print seq[i], "->", tar[i], ';', opt
      rec.step()


for i in range(10000):
   train_seq()

check()
