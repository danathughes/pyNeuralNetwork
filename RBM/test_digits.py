from digits import *
from RBM import *


def get_RBM():
   print "Loading digits...."
   digits, classes = load_digits('digits_train.txt')

   r = RBM(196,10)

   # Train at the following learning rates:

   print "Training..."

   learning_rates = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]

   for rate in learning_rates:
      print "Learning rate =", rate

      for i in range(20):
         print '  ' + str(i) + '...',
         r.train_epoch(digits, rate)
         err = r.free_energy(digits)
         print 'Energy = ' + str(err) + '...',
         PL = r.pseudolikelihood(digits)
         print 'Pseudolikelihood = ' + str(PL) + '...',
         print 'Done'

   return r, digits



