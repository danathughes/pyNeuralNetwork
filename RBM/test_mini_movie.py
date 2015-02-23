from RBM import *

def get_RBM():

   r = RBM(6,2)

   k = 9

   dataset = [[1,1,1,0,0,0],
              [1,0,1,0,0,0],
              [1,1,1,0,0,0],
              [0,0,1,1,1,0],
              [0,0,1,1,1,0],
              [0,0,1,1,1,0],
              [0,1,0,0,0,1],
              [0,1,0,0,0,1],
              [0,1,1,0,0,1]]

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
