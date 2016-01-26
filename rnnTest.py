from RecurrentNeuralNetwork.recurrentNeuralNetwork import *
from Functions.functions import *


# Make the same rnn
rnn = RecurrentNeuralNetwork([3,2,2],[None, LINEAR, SIGMOID])


# Set the weights the same
rnn.Wih[0,0] = 0.1
rnn.Wih[0,1] = 0.2
rnn.Wih[0,2] = 0.3
rnn.Wih[1,0] = 0.4
rnn.Wih[1,1] = 0.5
rnn.Wih[1,2] = 0.6

rnn.bh = 0*rnn.bh

rnn.Whh[0,0] = 0.1
rnn.Whh[0,1] = 0.2
rnn.Whh[1,0] = 0.2
rnn.Whh[1,1] = 0.1

rnn.Who[0,0] = 0.25
rnn.Who[0,1] = 0.75
rnn.Who[1,0] = 0.75
rnn.Who[1,1] = 0.25

rnn.bo = 0*rnn.bo

print rnn.Wih
print
print rnn.Whh
print
print rnn.Who
print
print rnn.bh
print
print rnn.bo
print

sequence = []
outputs = []

sequence.append(np.array([[0],[0],[0]]))
sequence.append(np.array([[0],[1],[0]]))
sequence.append(np.array([[0],[0],[1]]))
sequence.append(np.array([[1],[0],[1]]))

outputs.append(np.array([[1],[0]]))
outputs.append(np.array([[1],[0]]))
outputs.append(np.array([[0],[1]]))
outputs.append(np.array([[0],[1]]))

print "Activations"
print "==========="
activations = rnn.activate(sequence)

print
print "Deltas"
print "======"

old_hidden=np.zeros(rnn.bh.shape)

for t in range(len(activations)):
  act = activations[t]
  out = outputs[t]

  delta_out = rnn.cost_gradient(act[2], out)
  delta_out *= rnn.gradient_functions[2](act[2])

  delta_hidden = np.dot(rnn.Who.transpose(), delta_out)
  delta_hidden *= rnn.gradient_functions[1](act[1])

  delta_recurrent = np.dot(delta_hidden, old_hidden.transpose())
  old_hidden = act[1]

  print "Hidden:\t", delta_hidden.transpose()
  print "Recurrent:\t", delta_recurrent.transpose()
  print "Output:\t", delta_out.transpose()
  print



