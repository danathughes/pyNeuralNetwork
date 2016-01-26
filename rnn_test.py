import RecurrentNeuralNetwork.recurrentNeuralNetwork as RNN
import Training.teacher as Teacher
import Logger.consoleLogger as Logger
import numpy as np

rnn = RNN.RecurrentNeuralNetwork([2,4,1])
seq = []
out = []

seq.append(np.array([[0],[1]]))
seq.append(np.array([[0],[1]]))
seq.append(np.array([[0],[0]]))
seq.append(np.array([[1],[0]]))
seq.append(np.array([[1],[0]]))
seq.append(np.array([[1],[1]]))
seq.append(np.array([[1],[1]]))
seq.append(np.array([[0],[1]]))

out.append(np.array([[0]]))
out.append(np.array([[1]]))
out.append(np.array([[0]]))
out.append(np.array([[1]]))
out.append(np.array([[0]]))
out.append(np.array([[1]]))
out.append(np.array([[1]]))
out.append(np.array([[0]]))

l = Logger.ConsoleLogger(rnn,([seq],[out]),([seq],[out]))

T = Teacher.Teacher(rnn, l)
T.add_weight_update(-0.05, Teacher.gradient_descent)
#T.add_weight_update(0.1, Teacher.momentum)
#T.add_weight_update(0.0001, Teacher.weight_decay)

for i in range(100000):
  T.train([seq],[out])
  if i%500 == 0:
    l.log_training(i)


print rnn.predict(seq)
print out

