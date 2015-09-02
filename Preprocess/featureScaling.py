## featureScaling.py
##
## Schemes to scale the features of a dataset
##

import numpy as np

def mean_stdev(dataset):
   """
   Normalize the data by removing the mean and dividing by the standard
   deviation.
   """

   means = np.mean(dataset, 0)
   stdevs = np.std(dataset, 0)

   return (dataset - means) / stdevs


def min_max(data):
   """
   Normalize the data by scaling the min and max values between 0 and 1
   """

   mins = np.min(dataset,0)
   maxes = np.max(dataset,0)

   return (dataset - mins) / (maxes - mins)
