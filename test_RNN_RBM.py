from RNNRBM.RNNRBM import *
import numpy as np

# Simple RNN-RBM with 3 visible, 4 hidden and 2 rnn units
r = RNNRBM(3,4,2)

# Create a sample sequence
visible = [[0,0,1],[0,1,0],[1,0,0],[0,1,0],[0,0,1]]
visible_sequence = [np.array([v]).transpose() for v in visible]

# How many elements in the sequence?
N = len(visible_sequence)

print "Visible Sequence: "
for i in range(N):
   print i, '-'
   print visible_sequence[i]

print

initial_rnn = np.array([[0,0]]).transpose()

print "Initial RNN hidden layer: "
print initial_rnn
print

rnn_sequence = r.generate_rnn_sequence(visible_sequence, initial_rnn)

print "RNN Hidden Sequence: "
for i in range(N):
   print i, '-'
   print rnn_sequence[i]
print


bh_sequence = r.generate_bh_sequence(initial_rnn, rnn_sequence)
bv_sequence = r.generate_bv_sequence(initial_rnn, rnn_sequence)

print "Dynamic Hidden Bias Sequence: "
for i in range(N):
   print i, '-'
   print bh_sequence[i]
print

print "Dynamic Visible Bias Sequence: "
for i in range(N):
   print i, '-'
   print bv_sequence[i]
print


h0_sequence = []
hk_sequence = []
vk_sequence = []

k=1

for i in range(len(visible_sequence)):
   h0 = r.sample_hidden(visible_sequence[i], bh_sequence[i])
   vk, hk = r.block_gibbs(h0, bv_sequence[i], bh_sequence[i], k)
   h0_sequence.append(h0)
   hk_sequence.append(hk)
   vk_sequence.append(vk)

print "Visible-hidden pairs"
print '\tv0\t\th0\t\t\tvk\t\t\thk'

for i in range(N):
   print i,'-\t', visible_sequence[i].transpose().tolist(), 
   print '\t', h0_sequence[i].transpose().tolist(),
   print '\t', vk_sequence[i].transpose().tolist(),
   print '\t', hk_sequence[i].transpose().tolist()
print


print "Gradients -"
for i in range(N):
   bv = bv_sequence[i]
   bh = bh_sequence[i]
   v0 = visible_sequence[i]
   vk = vk_sequence[i]
   h0 = h0_sequence[i]
   hk = hk_sequence[i]

   print 'Time Step',i
   print '-----------'

   # Equation 13 and 15
   print 'dC/dbv(t)\t\tdC/dbh(t)'
   print (vk - v0).transpose().tolist(),
   bh_k = sigmoid(np.dot(r.Whv.transpose(), vk) - bh)
   bh_0 = sigmoid(np.dot(r.Whv.transpose(), v0) - bh)
   print '\t', (bh_k - bh_0).transpose().tolist()
   print

   # Equation 14
   print 'dC/dWhv(t)'
   Whv_k = np.dot(vk, sigmoid(np.dot(r.Whv.transpose(), vk) - bh).transpose())
   Whv_0 = np.dot(v0, sigmoid(np.dot(r.Whv.transpose(), v0) - bh).transpose())
   print Whv_k - Whv_0
   print

   # Equation 16
   print 'dC/dWuh(t)\t',
   rnn_prior = initial_rnn if i == 0 else rnn_sequence[i-1]
   print 'rnn =', rnn_prior.transpose().tolist()
   print np.dot((bh_k - bh_0), rnn_prior.transpose())
   print

   # Equation 17
   print 'dC/dWuv(t)'
   print np.dot((vk-v0), rnn_prior.transpose())
   print

   print
   print '--------'
   print

print

# Does the train_sequence function break?
r.train_sequence(visible_sequence, initial_rnn)


   
