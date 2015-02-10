## featureScaling.py
##
## Schemes to scale the features of a dataset
##

import numpy as np

def mean_stdev(data):
   """
   Normalize the data by removing the mean and dividing by the standard
   deviation.
   """

   data_array = np.array(data)
   means = np.zeros(data_array.shape[1])
   stdevs = np.zeros(data_array.shape[1])

   for i in range(data_array.shape[1]):
      means[i] = np.mean(data_array[:,i])
      stdevs[i] = np.std(data_array[:,i])

   for i in range(len(data)):
      for j in range(len(data[0])):
         data[i][j] = (data[i][j] - means[j])/stdevs[j]

   return data


def min_max(data):
   """
   Normalize the data by scaling the min and max values between 0 and 1
   """

   data_array = np.array(data)
   mins = np.zeros(data_array.shape[1])
   maxes = np.zeros(data_array.shape[1])

   for i in range(data_array.shape[1]):
      mins[i] = np.min(data_array[:,i])
      maxes[i] = np.max(data_array[:,i])

   for i in range(len(data)):
      for j in range(len(data[0])):
         data[i][j] = (data[i][j] - mins[j])/(maxes[j] - mins[j])

   return data
