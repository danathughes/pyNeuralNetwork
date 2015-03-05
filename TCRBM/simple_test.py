import numpy as np
import matplotlib.pyplot as plt
from TRBM import *


# Make a random dataset
seq = [[0,0],[0,1],[0,1],[1,1],[1,0],[1,0]]
seq = seq * 100

# Make a CRBM
print "Creating RBM"

rbm = TRBM(2,5,3)

# Train it
k=1
r = 0.9
for i in range(10000):
   if i%10 == 0:
      print "Epoch", i
      rbm.train_epoch(seq, r, k)

   if i%1000 == 0:
      k += 1
      r -= 0.05

# Time to plot!
pred = seq[0:4]

for i in range(4,len(seq)):
   vis = [np.array([s]).transpose() for s in seq[i-3:i]]
   v0 = np.array([seq[4]]).transpose()
   # Predict h
   h = rbm.sample_hidden(v0, vis)
   # Predict v
   vis = vis[1:] + [v0]
   v = rbm.sample_visible(h, vis)
   pred.append([v[0,0], v[1,0]])

X = []
Y = []

Xp = []
Yp = []

for i in range(len(seq)):
   X.append(seq[i][0])
   Y.append(seq[i][1] + 1.1)
   Xp.append(pred[i][0] + 2.5)
   Yp.append(pred[i][1] + 3.6)

x = range(len(seq))

plt.plot(x, X, '-b', x, Y, '-r', x, Xp, '-b', x, Yp, '-r')
plt.show()
