## predictHousingCosts.py
##
## Simple script to predict iris

from logisticRegression import *
import random
import matplotlib.pyplot as plt


irisDataFileName = 'iris.data'

training_percentage = 0.8


def normalize(data):
   """
   """

   data_array = np.array(data)
   #create a zero vector [1x4] - means for each parameters N.  numpy.Shape function outputs dim of array (n,m) so shape[1] outputs the col length
   means = np.zeros(data_array.shape[1])
   stdevs = np.zeros(data_array.shape[1])

   #calculate the means and standard deviation for entire dataset for each column
   for i in range(data_array.shape[1]):
      means[i] = np.mean(data_array[:,i])
      stdevs[i] = np.std(data_array[:,i])
   #normalize each value to be between -1 and 1
   for i in range(len(data)):
      for j in range(len(data[0])):
         data[i][j] = (data[i][j] - means[j])/stdevs[j]

   return data


if __name__ == '__main__':
   # Load the data

   iris_data_file = open(irisDataFileName)
   iris_data = []

   for data in iris_data_file.readlines():
      #create a temp vector with all of the columns separated
      tmp = [eval(num) for num in data.split(',')[0:4]]
      #for each line save the flower name into a separate column
      flower = data.split(',')[-1].strip()
      if flower == 'Iris-setosa':
         tmp.append([1,0,0])
      elif flower == 'Iris-versicolor':
         tmp.append([0,1,0])
      else:
         tmp.append([0,0,1])
      iris_data.append(tmp)


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

   # Normalize the input data
   training_set_X = normalize(training_set_X)
   test_set_X = normalize(test_set_X)

   # How many variables?
   numVariables = len(training_set_X[0])

   # Create the model
   LR = LogisticRegressionModel(numVariables, 3)

   # Train the model
   LR.train_minibatch(training_set_X, training_set_Y, 0.1, 0.01, 1000)


   # How'd we do?
   np.set_printoptions(precision=5)
   np.set_printoptions(suppress=True)

   print "Weights:"
   print "========"
   print LR.weights

   print
   print "Total Cost -", LR.cost(training_set_X, training_set_Y)


   print
   print "Training Results"
   print "================"
   total_training_error = 0.0
   print "   Run         Predicted Probability           Actual Result    Error?"
   for i in range(len(training_set_X)):
      prediction = LR.predict(training_set_X[i])
      total_training_error += abs(training_set_Y[i] - prediction)
      #create error vector with length of # of parameters
      error=np.zeros(len(training_set_Y[i]))
      #highest index prediction value will be set to 1
      error[np.argmax(error)]=1
      #if prediction does not match actual selection
      errormsg = ''
      if np.sum(error - training_set_Y[i]) != 0:
            errormsg = 'yes'
      print "  ", i, "-", prediction, "\t", training_set_Y[i], errormsg
   mean_training_error = total_training_error / len(training_set_X)

   print
   print "Test Results"
   print "============"
   total_test_error = 0.0
   for i in range(len(test_set_X)):
      prediction = LR.predict(test_set_X[i])
      total_test_error += abs(test_set_Y[i] - prediction)
      print "  ", i, "-", prediction, "\t", test_set_Y[i]
   mean_test_error = total_test_error / len(test_set_X)

   print
   print "Total Errors"
   print "============"
   print "Total Training Error:", np.sum(total_training_error)
   print "Mean Training Error: ", np.mean(mean_training_error)
   print "Total Test Error:    ", np.sum(total_test_error)
   print "Mean Test Error:     ", np.mean(mean_test_error)

   print
   a=raw_input('What is the name of the first classification:\t')
   b=raw_input('What is the name of the second classification:\t')
   c=raw_input('What is the name of the third classification:\t')

#   d=1
#   e=200
#   f=30
#   print
#   print '==Confusion Matrix=='
#   print
#   print 'a b c  <-- classified as'
#   print '%d %d %d | a = %s' %(d, e,f,a)
#   print '%d %d %d | b = %s' %(d, e,f,b)
#   print '%d %d %d | c = %s' %(d, e,f,c)

#   x_vals = np.zeros(len(test_set_X))
#   predictions = np.zeros((len(test_set_Y),2))
#   for i in range(len(test_set_X)):
#      predictions[i,1] = LR.predict(test_set_X[i])
#      predictions[i,0] = test_set_Y[i]
#      x_vals[i] = i+1

#   plt.plot(x_vals, predictions[:,0], x_vals, predictions[:,1])
#   plt.show()


