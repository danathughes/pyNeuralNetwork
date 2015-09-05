from nn.components.layers.RecurrentLayer import *
from nn.components.layers.SigmoidLayer import *
from nn.components.layers.InputLayer import *
from nn.components.connections.FullConnection import *
from nn.components.connections.Bias import *
from nn.components.objectives.MSEObjective import *
from nn.models.RecurrentNeuralNetwork import *
from trainers.SGD import SGDTrainer

import numpy as np

# Build a network

net = RecurrentNeuralNetwork()

inp = InputLayer(1)
rec = RecurrentLayer(5)
out = SigmoidLayer(1)

rec.zeroInitialHistoryBatch(2)

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

net.addLayer(inp)
net.addLayer(tgt)
net.addConnection(i2r)
net.addConnection(rbias)
net.addRecurrentLayer(rec)
net.addConnection(r2o)
net.addConnection(obias)
net.addLayer(out)
net.addObjective(obj)

net.setInputLayer(inp)
net.setTargetLayer(tgt)
net.setOutputLayer(out)
net.setObjective(obj)

sgd = SGDTrainer(net, learning_rate=0.1, momentum = 0.9)


seq = np.array([[[1],[1],[0]],[[1],[0],[0]],[[1],[1],[0]],[[0],[0],[0]],[[0],[1],[0]],[[0],[0],[0]],[[1],[1],[0]],[[1],[0],[0]],[[1],[1],[0]],[[0],[0],[0]],[[0],[1],[0]],[[0],[0],[0]],[[1],[1],[0]],[[1],[0],[0]],[[1],[1],[0]],[[0],[0],[0]],[[0],[1],[0]],[[0],[0],[0]],[[1],[1],[0]],[[1],[0],[0]],[[1],[1],[0]],[[0],[0],[0]],[[0],[1],[0]],[[0],[0],[0]]])
tar = np.array([[[1],[1],[0]],[[0],[1],[0]],[[1],[0],[0]],[[1],[0],[0]],[[1],[1],[0]],[[1],[1],[0]],[[0],[0],[0]],[[1],[0],[0]],[[0],[1],[0]],[[0],[1],[0]],[[0],[0],[0]],[[0],[0],[0]],[[1],[1],[0]],[[0],[1],[0]],[[1],[0],[0]],[[1],[0],[0]],[[1],[1],[0]],[[1],[1],[0]],[[0],[0],[0]],[[1],[0],[0]],[[0],[1],[0]],[[0],[1],[0]],[[0],[0],[0]],[[0],[0],[0]]])

dataset = (seq, tar)

def train_seq(t):
   print "   t:", t, "\tObj = ", net.getSequenceObjective(dataset)
   sgd.trainBatch(dataset)


def check():
   net.reset()
   for i in range(24):
      net.setInput(dataset[0][i,:])
      net.forward()
      opt = net.getOutput()
      print dataset[0][i].transpose(), "->", dataset[1][i].transpose(), ';', opt.transpose()
      net.step()


for i in range(10000):
   train_seq(i)

check()
