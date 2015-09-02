## predictAutoMPG.py
##
## Simple script to predict gas milage on automobiles

from linearRegression2 import *
import random
import matplotlib.pyplot as plt


autoDataFileName = 'auto-mpg.data'

training_percentage = 0.8

# normalize function data(in this case the variables of the dataset)
def normalize(data):
   """
   """

   data_array = np.array(data)
   #create means zero vector as the same size as data
   means = np.zeros(data_array.shape[1])
   #create stdev zero vector as the same size as data
   stdevs = np.zeros(data_array.shape[1])

   for i in range(data_array.shape[1]):
      #for each index of mean and stdevs, take the mean/std of values in data_array's column index
      means[i] = np.mean(data_array[:,i])
      stdevs[i] = np.std(data_array[:,i])
   # vectorize (number of training data)
   for i in range(len(data)):
     # vectorize (number of parameters in each training data)
      for j in range(len(data[0])):
         # normalize each variable in the data set to be between -1 and 1
         data[i][j] = (data[i][j] - means[j])/stdevs[j]

   return data

# following code will only run if the .py file is being run and not when just importing it.
if __name__ == '__main__':
   # Load the data

   auto_data_file = open(autoDataFileName)

   auto_data = []

   #for each read line of the data
   for data in auto_data_file.readlines():
      #split each line by index 0-8, and evaluate and append each index in to autodata[]
      auto_data.append([eval(num) for num in data.split()[0:8]])

   # Split into training and test data
   training_set_X = []
   training_set_Y = []
   test_set_X = []
   test_set_Y = []

   for item in auto_data:
      # randomly pick data for training
      if random.random() < training_percentage:
         #append index 1 to end of vector to X
         training_set_X.append(item[1:])
         #append index 0 to Y
         training_set_Y.append(item[0])
      else:
         # if it was not randomly picked as training set, set it as the test set
         test_set_X.append(item[1:])
         test_set_Y.append(item[0])

   #normalize training and testing set X(variables)
   training_set_X = normalize(training_set_X)
   test_set_X = normalize(test_set_X)

   # How many variables?
   numVariables = len(training_set_X[0])

   # Create the model
   LR = LinearRegressionModel(numVariables)

   # Train the model
   LR.train_batch(training_set_X, training_set_Y, 0.7, 100.0, 200)  #LR calls train batch function using (training set data, output, learning_rate, convergence, maxEpochs) for parameters

   print "Training Method Used:"
   print "========"
   print ## code that prints out module used in the training.


   # How'd we do?
   print "Weights:"
   print "========"
   # return weights from Training method
   print LR.weights

   print
   # execute LR model, call cost function using training data and training output
   print "Total Cost -", LR.cost(training_set_X, training_set_Y)


   print
   print "Training Results"
   print "================"
   total_training_error = 0.0
   for i in range(len(training_set_X)):
      # make a prediction for each training set
      prediction = LR.predict(training_set_X[i])
      total_training_error += abs(training_set_Y[i] - prediction)
      print "  ", i, "-", prediction, "\t", training_set_Y[i]
   mean_training_error = total_training_error / len(training_set_X)

   print
   print "Test Results"
   print "============"
   total_test_error = 0.0
   for i in range(len(test_set_X)):
      # make a prediction for each test set
      prediction = LR.predict(test_set_X[i])
      total_test_error += abs(test_set_Y[i] - prediction)
      print "  ", i, "-", prediction, "\t", test_set_Y[i]
   mean_test_error = total_test_error / len(test_set_X)

   print
   print "Total Errors"
   print "============"
   print "Total Training Error:", total_training_error
   print "Mean Training Error: ", mean_training_error
   print "Total Test Error:    ", total_test_error
   print "Mean Test Error:     ", mean_test_error


   x_vals = np.zeros(len(test_set_X))
   # create a zero vector rows=# of test data, 2 columns
   predictions = np.zeros((len(test_set_Y),2))
   for i in range(len(test_set_X)):
      #predictions vector index 1 is the LR prediction
      predictions[i,1] = LR.predict(test_set_X[i])
      #predictions vector index 0 is the actual test set output
      predictions[i,0] = test_set_Y[i]
      #shift each iteration of x_val by 1 (so it can be indexed from 1 to Z)
      x_vals[i] = i+1

   print
   print "Plotting Information"
   print "============"
   yaxislabel=raw_input('What is the predicted output (y-label)?')
   plt.ylabel(yaxislabel)
   graphtitle=raw_input('What is the plot title?')
   plt.title(graphtitle)
   plt.xlabel('Test Samples')
   plt.plot(x_vals, predictions[:,0], x_vals, predictions[:,1])
   plt.show()


