from digits import *
from GB_RBM import *


def get_RBM():
   print "Loading digits...."
   digits, classes = load_digits('digits_train.txt')
#   digits = [digit_gray_to_binary(d) for d in digits]
   r = GB_RBM(196,25)

   r.bias_visible = np.mean(np.array([digits]),1).transpose()

   print r.bias_visible

   raw_input()


   # Train at the following learning rates:
   print "Training..."

   learning_rates = [0.1, 0.05, 0.01, 0.005, 0.001]

   for rate in learning_rates:
      print "Learning rate =", rate

      for i in range(20):
         print '  ' + str(i) + '...',
         r.train_epoch(digits, rate, 5)
         err = np.sum([r.free_energy(digit) for digit in digits])
         print 'Energy = ' + str(err) + '...',
         PL = r.pseudolikelihood(digits)
         print 'Pseudolikelihood = ' + str(PL) + '...',
         print 'Done'

   return r, digits



