## predictHousingCosts.py
##
## Simple script to predict iris

from neuralNetwork import *
import random
import matplotlib.pyplot as plt


irisDataFileName = 'iris.data'

training_percentage = 1.0

if __name__ == '__main__':
   # Load the data

   iris_data_file = open(irisDataFileName)
   iris_data = []

   for data in iris_data_file.readlines():
      tmp = [eval(num) for num in data.split(',')[0:4]]
      flower = data.split(',')[-1].strip()
      if flower == 'Iris-setosa':
         tmp.append([1,0,0])
      elif flower == 'Iris-versicolor':
         tmp.append([0,1,0])
      else:
         tmp.append([0,0,1])
      iris_data.append(tmp)


   # Perform mean normalization on the non-output values
   data_array = np.zeros((len(iris_data), len(iris_data[0])-1))
   for i in range(len(iris_data)):
      for j in range(len(iris_data[0])-1):
         data_array[i,j] = iris_data[i][j]

   means = np.zeros(data_array.shape[1])
   stdevs = np.zeros(data_array.shape[1])

 
   for i in range(data_array.shape[1]):
      means[i] = np.mean(data_array[:,i])
      stdevs[i] = np.std(data_array[:,i])

   for i in range(len(iris_data)):
      for j in range(len(iris_data[0])-1):
         iris_data[i][j] = (iris_data[i][j] - means[j])/stdevs[j]


   # Split into training and test data
   training_set_X = []
   training_set_Y = []
   test_set_X = []
   test_set_Y = []

   for item in iris_data:
      if random.random() < training_percentage:
         training_set_X.append(item[:-1])
         training_set_Y.append(item[-1])
      else:
         test_set_X.append(item[:-1])
         test_set_Y.append(item[-1])

   # How many variables?
   numVariables = len(training_set_X[0])

   # Create the model
   NN = NeuralNetwork([numVariables, 4, 3], [None, SIGMOID, SIGMOID])

   # Train the model
   NN.train_batch(training_set_X, training_set_Y, 1, 0.001, 2, 0.0001)

   # How'd we do?
#   print "Weights:"
#   print "========"
#   print NN.weights

   np.set_printoptions(precision=5)
   np.set_printoptions(suppress=True)

#   print
#   print "Total Cost -", NN.cost(training_set_X, training_set_Y, 0.0001)


#   print
#   print "Training Results"
#   print "================"
#   total_training_error = 0.0
#   for i in range(len(training_set_X)):
#      prediction = NN.predict(training_set_X[i])
#      total_training_error += abs(training_set_Y[i] - prediction)
#      print "  ", i, "-", prediction, "\t", training_set_Y[i]
#   mean_training_error = total_training_error / len(training_set_X)

#   print
#   print "Test Results"
#   print "============"
#   total_test_error = 0.0
#   for i in range(len(test_set_X)):
#      prediction = NN.predict(test_set_X[i])
#      total_test_error += abs(test_set_Y[i] - prediction)
#      print "  ", i, "-", prediction, "\t", test_set_Y[i]
#   mean_test_error = total_test_error / len(test_set_X)

#   print
#   print "Total Errors"
#   print "============"
#   print "Total Training Error:", np.sum(total_training_error)
#   print "Mean Training Error: ", np.mean(mean_training_error)
#   print "Total Test Error:    ", np.sum(total_test_error)
#   print "Mean Test Error:     ", np.mean(mean_test_error)

#   x_vals = np.zeros(len(test_set_X))
#   predictions = np.zeros((len(test_set_Y),2))
#   for i in range(len(test_set_X)):
#      predictions[i,1] = NN.predict(test_set_X[i])
#      predictions[i,0] = test_set_Y[i]
#      x_vals[i] = i+1

#   plt.plot(x_vals, predictions[:,0], x_vals, predictions[:,1])
#   plt.show()


