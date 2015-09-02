from RNNRBM_GB.RNNRBM import *
import numpy as np

import matplotlib.pyplot as plt
from Training.teacher import *

# Simple RNN-RBM with 3 visible, 4 hidden and 2 rnn units
r = RNNRBM(3,10,10)


# Create a sample sequence
#visible = [[0.0, 0.0, 1.0],
#           [0.0, 1.0, 0.0],
#           [1.0, 0.0, 0.0],
#           [1.0, 1.0, 0.0],
#           [1.0, 0.0, 1.0]]

# Create a sample sequence
visible = [[-1.001, -0.805,  1.670],
           [-1.194,  1.757, -0.592],
           [ 1.502, -0.622, -0.765],
           [ 0.539,  0.476, -0.939],
           [ 0.154, -0.805,  0.626]]
visible_sequence = [np.array([v]).transpose() for v in visible]

# How many elements in the sequence?
N = len(visible_sequence)

print "Visible Sequence: "
for i in range(N):
   print i, '-'
   print visible_sequence[i]

print

gradient_descent_rate = 0.2

teacher = Teacher(r)
teacher.add_weight_update(gradient_descent_rate, gradient_descent)

# Does the train_sequence function break?
l = -0.2
ground_truth = np.array(visible)



for i in range(1,10000):

   if i%500 == 0:
      gradient_descent_rate = gradient_descent_rate / 2
      teacher.weight_updates = []
      teacher.update_rates = []
      teacher.add_weight_update(gradient_descent_rate, gradient_descent)

   print 'Iteration', i,

   teacher.train([visible_sequence], [])

#   dWhv, dWuh, dWuv, dWuu, dWvu, dbv, dbh, dbu, du0 = r.gradient([visible_sequence])
#   r.update_weights(l*dWhv, l*dWuh, l*dWuv, l*dWuu, l*dWvu, l*dbv, l*dbh, l*dbu, l*du0)


   v_gen = []
   v_guess = np.zeros((3,1))
   prior_rnn = r.initial_rnn
   for i in range(N):
      v_next = r.generate_visible(prior_rnn, v_guess, 20)
      prior_rnn = r.get_rnn(v_next, prior_rnn)
      v_gen.append(v_next)
      v_guess = v_next


   M = 10 # Number of samples to figure out statistics
   # Get some samples to figure out the mean and std-dev of reconstructed signals
   samples = [np.zeros((3,M)), np.zeros((3,M)), np.zeros((3,M)), np.zeros((3,M)), np.zeros((3,M))]
   cost = 0.0
   for i in range(M):
      prior_rnn = r.initial_rnn
      v_guess = np.zeros((3,1))
      for j in range(N):
         v_next = r.generate_visible(prior_rnn, v_guess, 20)
         v_guess = np.array([visible[j]]).transpose()
         prior_rnn = r.get_rnn(v_guess, prior_rnn)
         cost = cost + np.sum((v_next - np.array(visible[j]))**2)

         samples[j][0,i]=v_next[0,0]
         samples[j][1,i]=v_next[1,0]
         samples[j][2,i]=v_next[2,0]
       

   means = [np.mean(samples[0],1), np.mean(samples[1],1), np.mean(samples[2],1), np.mean(samples[3],1), np.mean(samples[4],1)]
   stds = [np.std(samples[0],1), np.std(samples[1],1), np.std(samples[2],1), np.std(samples[3],1), np.std(samples[4],1)]

   mean_0 = [means[0][0], means[1][0], means[2][0], means[3][0], means[4][0]]
   mean_1 = [means[0][1], means[1][1], means[2][1], means[3][1], means[4][1]]
   mean_2 = [means[0][2], means[1][2], means[2][2], means[3][2], means[4][2]]

   std_0 = [stds[0][0], stds[1][0], stds[2][0], stds[3][0], stds[4][0]]
   std_1 = [stds[0][1], stds[1][1], stds[2][1], stds[3][1], stds[4][1]]
   std_2 = [stds[0][2], stds[1][2], stds[2][2], stds[3][2], stds[4][2]]

   hi_0 = [mean_0[0] + std_0[0], mean_0[1] + std_0[1], mean_0[2] + std_0[2], mean_0[3] + std_0[3], mean_0[4] + std_0[4]]
   hi_1 = [mean_1[0] + std_1[0], mean_1[1] + std_1[1], mean_1[2] + std_1[2], mean_1[3] + std_1[3], mean_1[4] + std_1[4]]
   hi_2 = [mean_2[0] + std_2[0], mean_2[1] + std_2[1], mean_2[2] + std_2[2], mean_2[3] + std_2[3], mean_2[4] + std_2[4]]

   lo_0 = [mean_0[0] - std_0[0], mean_0[1] - std_0[1], mean_0[2] - std_0[2], mean_0[3] - std_0[3], mean_0[4] - std_0[4]]
   lo_1 = [mean_1[0] - std_1[0], mean_1[1] - std_1[1], mean_1[2] - std_1[2], mean_1[3] - std_1[3], mean_1[4] - std_1[4]]
   lo_2 = [mean_2[0] - std_2[0], mean_2[1] - std_2[1], mean_2[2] - std_2[2], mean_2[3] - std_2[3], mean_2[4] - std_2[4]]

   f = plt.figure(1)
   plt.hold(False)
   plt.plot([1,2,3,4,5], ground_truth[:,0], '-r', linewidth=2)
   plt.hold(True)
   plt.plot([1,2,3,4,5], ground_truth[:,1], '-g', linewidth=2)
   plt.plot([1,2,3,4,5], ground_truth[:,2], '-b', linewidth=2)
   plt.plot([1,2,3,4,5], mean_0, '-r')
   plt.plot([1,2,3,4,5], mean_1, '-g')
   plt.plot([1,2,3,4,5], mean_2, '-b')
  
   plt.plot([1,2,3,4,5], hi_0, '--r')
   plt.plot([1,2,3,4,5], hi_1, '--g')
   plt.plot([1,2,3,4,5], hi_2, '--b')

   plt.plot([1,2,3,4,5], lo_0, '--r')
   plt.plot([1,2,3,4,5], lo_1, '--g')
   plt.plot([1,2,3,4,5], lo_2, '--b')

   plt.axis([1,5,-2,2.5])

   plt.draw()
   f.show()

   print 'RMS error = ', r.cost([visible], [], 10, 20)

#   if i%1000 == 0:
#      l = l + 0.05
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
