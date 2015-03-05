from CRBM import *
import numpy as np

def get_CRBM(how_long = 4000):
   print "Creating data...."
   mean1 = np.array([-0.2, 0.3])
   mean2 = np.array([0.5, -0.5])

   cov1 = np.array([[0.02, 0.005],[0.001, 0.01]])
   cov2 = np.array([[0.02, 0.0],[0.0, 0.02]])
   
   dataset = np.random.multivariate_normal(mean1, cov1, 200).tolist()
   dataset += np.random.multivariate_normal(mean2, cov2, 200).tolist()

   r = CRBM(2,4)

   # Train at the following learning rates:
   print "Training..."

   for i in range(how_long):
      print '  ' + str(i) + '...',
      r.train_epoch(dataset, (0.9, 0.9), 20)
      E = np.sum([r.energy(data) for data in dataset])
      print 'Energy = ' + str(E) + '...',
      print 'Done'

   return r, dataset



