import numpy as np
import matplotlib.pyplot as plt
from CRBM import *


# Make a random dataset
#dataset = [[np.random.normal(1,0.2), np.random.normal(1,0.1)] for i in range(50)]
#dataset = dataset + [[np.random.normal(-1,0.2), np.random.normal(-1,0.1)] for i in range(50)]

# This one's really quite hard...
dataset = [[np.sin(2.5*np.pi*n/180) + np.random.normal(0,0.05), np.cos(2.5*np.pi*n/180) + np.random.normal(0,0.15)] for n in range(100)]

#dataset = [[-1.0 + np.random.normal(0,0.15) + 0.02*n, 0.5 + np.random.normal(0,0.2)] for n in range(100)]

XY = np.array(dataset)

# And show it
f = plt.figure(1)
plt.scatter(XY[:,0], XY[:,1], 20, 'b', 'o')
plt.title('Distribution')
plt.draw()
f.show()

# Make a CRBM
print "Creating RBM"

rbm = CRBM(2,20)

rbm.lo = -2
rbm.hi = 2

# Train it
k=20
for i in range(10000):
#for i in range(0):
   err = rbm.train_epoch(dataset, 0.1, k)
   if i%10 == 0:
      print "Epoch", i, ": Error =", err

      # Now reconstruct some stuff
      XY_rec = np.random.uniform(-2,2,(100,2))

      for i in range(100):
         v0 = np.array([XY_rec[i,:]]).transpose()
         for j in range(20):
            h0 = rbm.sample_hidden(v0)
            v0 = rbm.sample_visible(h0)
         XY_rec[i,0] = v0[0,0]
         XY_rec[i,1] = v0[1,0]

      # Finally, draw the sampled points
      plt.hold(False)
      plt.scatter(XY[:,0], XY[:,1], 20, 'b', 'o')
      plt.title('Distribution')
      plt.hold(True)
      plt.scatter(XY_rec[:,0], XY_rec[:,1], 20, 'r', '+')
      plt.axis([-2,2,-2,2])
      plt.draw()
      f.show()


plt.show()

# Display the CRBM Parameters
print "CRBM"
print "===="
print
print "Weights:"
print "--------"
print rbm.weights
print
print "Visible Biases:"
print "---------------"
print rbm.bias_visible
print
print "Hidden Biases:"
print "--------------"
print rbm.bias_hidden
print
print "A Visible"
print "---------"
print rbm.A_visible
print
print "A Hidden"
print "--------"
print rbm.A_hidden
print

