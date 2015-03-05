from GB_RBM import *

def get_RBM():

   r = GB_RBM(6,3)

   k = 9

   dataset = [[ 1.485,  0.537,  0.374, -0.786, -0.775, -0.547],
              [ 1.360,  0.143,  0.689, -0.763, -0.709, -0.672],
              [ 1.385,  0.590,  0.346, -0.694, -0.554, -0.772],
              [-0.768, -1.486,  0.631,  1.457,  1.299, -0.572],
              [-0.542, -1.355,  0.289,  1.572,  1.432, -0.822],
              [-0.668, -1.250,  0.346,  1.183,  1.500, -0.797],
              [-0.743,  0.879, -1.653, -0.534, -0.753,  1.327],
              [-0.718,  1.168, -1.996, -0.649, -0.664,  1.127],
              [-0.793,  0.774,  0.974, -0.786, -0.775,  1.727]]

   # Train at the following learning rates:

   print "Training..."

   learning_rates = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
#   learning_rates = [0.1]

   for rate in learning_rates:
      print "Learning rate =", rate

      for i in range(100):
         print '  ' + str(i) + '...',
         r.train_epoch(dataset, rate, k)
         err = np.sum([r.free_energy(data) for data in dataset])
         print 'Energy = ' + str(err) + '...',
         PL = r.pseudolikelihood(dataset)
         print 'Pseudolikelihood = ' + str(PL) + '...',
         L = r.likelihood(dataset)
         print 'Likelihood = ' + str(L) + '...',
         print 'Done'


   return r, dataset
