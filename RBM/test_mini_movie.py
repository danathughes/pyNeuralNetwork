from RBM import *

def get_RBM():

   r = RBM(6,2)

   k = 9

   dataset = [[1,1,1,0,0,0],
              [1,0,1,0,0,0],
              [1,1,1,0,0,0],
              [0,0,1,1,1,0],
              [0,0,1,1,1,0],
              [0,0,1,1,1,0]]

   # Train at the following learning rates:

   print "Training..."

   learning_rates = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
#   learning_rates = [0.1]

   for rate in learning_rates:
      print "Learning rate =", rate

      for i in range(20):
         print '  ' + str(i) + '...',
         r.train_epoch(dataset, rate, k)
         print 'Done'


   return r, dataset
