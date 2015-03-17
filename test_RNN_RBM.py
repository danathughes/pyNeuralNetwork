from RNNRBM.RNNRBM import *
import numpy as np

# Simple RNN-RBM with 3 visible, 4 hidden and 2 rnn units
r = RNNRBM(3,8,4)

# Create a sample sequence
visible = [[0,0,1],[0,1,0],[1,0,0],[1,1,0],[1,0,1]]
visible_sequence = [np.array([v]).transpose() for v in visible]

# How many elements in the sequence?
N = len(visible_sequence)

print "Visible Sequence: "
for i in range(N):
   print i, '-'
   print visible_sequence[i]

print

initial_rnn = np.array([[0,0,0,0]]).transpose()

print "Initial RNN hidden layer: "
print initial_rnn
print


# Does the train_sequence function break?
l = -0.55

for i in range(1000):
   print 'Iteration', i
   dWhv, dWuh, dWuv, dWuu, dWvu, dbv, dbh, dbu, du0 = r.train_sequence(visible_sequence, initial_rnn)

   r.update_weights(l*dWhv, l*dWuh, l*dWuv, l*dWuu, l*dWvu, l*dbv, l*dbh, l*dbu)
   initial_rnn = initial_rnn + l*du0

   v_gen = []
   v_guess = np.zeros((3,1))
   prior_rnn = initial_rnn
   for i in range(N):
      v_next = r.generate_visible(prior_rnn, v_guess, 20)
      prior_rnn = r.get_rnn(v_next, prior_rnn)
      v_gen.append(v_next)
      v_guess = v_next

   for j in range(N):
      print j, '-', v_gen[j].transpose().tolist()

   if i%1000 == 0:
      l = l + 0.05
print

print 'Whv'
print r.Whv
print

print 'Wuh'
print r.Wuh
print

print 'Wuv'
print r.Wuv
print

print 'Wuu'
print r.Wuu
print

print 'Wvu'
print r.Wvu
print

print 'bias visible -',
print r.bias_visible.transpose().tolist()
print

print 'bias hidden -',
print r.bias_hidden.transpose().tolist()
print

print 'bias rnn -',
print r.bias_rnn.transpose().tolist()
print
