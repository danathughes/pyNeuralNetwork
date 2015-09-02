## linearRegression.py    Dana Hughes    version 1.01     17-November-2014
##
## Linear Regression model, which predicts a single output value given an 
## input vector.
##
## Revisions:
##   1.01  06-Dec-2014    Replaced loops with numpy vector operations where possible
##   1.0   17-Nov-2014    Initial version, consisting of a LinearRegressionModel 
##                        with batch and stochastic gradient descent.

import numpy as np
import random
import copy

class LinearRegressionModel:                                                  ##JC Notes:  Initiation of a new class object
   """
   """

   def __init__(self, numVariables):
      """
      """

      # Set the weights to zero, including an extra weight as the offset
      self.N = numVariables                                                   ## New class attribute N - number of variables
      ## New class attribute weights(coefficients) # of variables + 1 for the offset No
      self.weights = np.zeros(numVariables + 1)
      ## assign random weights
      for i in range(self.N):
          self.weights[i] = random.random()
     
   #cost function taking data: X training data (variables) output: Y training data
   def cost(self, data, output):
      """
      """

      # Add the offset term to the data
      cost = 0.0

      #for entire training data set calculate the running cost= square error
      for i in range(len(data)):
         # predict ( sum(random weights * data set's variables))
         prediction = self.predict(data[i])
         ## calculate error for each prediction
         error = prediction - output[i]
         cost = cost + error**2
      #linear regression cost function J(O) =1/2m * sum(h(x)-y)^2 .
      return cost/2/len(data)


   def gradient(self, data, output):
      """
      Return the gradient of the cost function wrt weights given the data and
      ground truth
      """
      # create a null vector based on # of variables + 1 for offset term
      gradient = np.zeros(self.N + 1)
      # create a vector of predicted values (weights * feature values)
      prediction = self.predict_vector(np.array(data))
      errors = prediction - np.array(output)
      gradient[0] = np.sum(errors)

      # for each feature
      for j in range(self.N):
         #j+1 to shift the index since grad[0] has already been set
         gradient[j+1] = 0
         #for each training set
         for i in range(len(errors)):
            #running total adding gradient for each row of each column
            gradient[j+1] = gradient[j+1] + data[i][j] * errors[i]
         gradient[j+1] = gradient[j+1]
      #returns gradient vector of errors for each feature [1x8]
      return gradient


   def train_batch(self, data, output, learning_rate = 0.1, convergence = 0.0001, maxEpochs = 10000):
      """
      """
      #epochs= # of times gradient descent runs
      epoch = 0

      while self.cost(data, output) > convergence and epoch < maxEpochs:
         print "Epoch", epoch, "- Cost:", self.cost(data,output)
         epoch+=1
         gradient = np.array(self.gradient(data, output))
         #update weights = theta - learning rate *  1/m*sum of errors
         self.weights -= learning_rate * gradient / len(data)


   def train_batch_backtrack_line_search(self, data, output, B=0.8, convergence = 0.0001, maxEpochs = 10000):
      """
      Perform full batch gradient descent with line search
      """

      epoch = 0

      while self.cost(data, output) > convergence and epoch < maxEpochs:
         print "Epoch", epoch, "- Cost:", self.cost(data,output)
         epoch+=1
         gradient = self.gradient(data, output)
         
         current_cost = self.cost(data,output)

         self.weights -= gradient
         full_step_cost = self.cost(data,output)
         self.weights += gradient

         t=1.0
         maxSteps = 20
         step = 0

         while (full_step_cost > current_cost - (t/2)*(sum(gradient**2))) and (step < maxSteps):
            t = B*t
            step += 1
         
         self.weights -= t*gradient/len(data)


   def train_minibatch(self, data, output, learning_rate = 0.1, convergence = 0.0001, maxEpochs = 10000):
      """
      """

      epoch = 0

      while self.cost(data, output) > convergence and epoch < maxEpochs:
         print "Epoch", epoch, "- Cost:", self.cost(data,output)
         epoch+=1
         size=int(np.floor(len(data)/10))
         for k in range(10):
             subdata = data[size*k:size*(k+1)]
             suboutput = output[size*k:size*(k+1)]
             gradient = np.array(self.gradient(subdata, suboutput))
             for i in range(self.N + 1):
                self.weights[i] -= learning_rate * gradient[i] / len(subdata)


   def train_stochastic(self, data, output, learning_rate = 0.1, convergence = 0.0001, maxEpochs = 10000):
      """
      """

      epoch = 0

      while self.cost(data, output) > convergence and epoch < maxEpochs:
         print "Epoch", epoch, "- Cost:", self.cost(data,output)
         epoch+=1
         for i in range(len(data)):
            error = output[i] - self.predict(data[i])
            self.weights[0] += learning_rate * error
            for j in range(self.N):
               self.weights[j+1] -= learning_rate * error * data[i][j]
 

   # Predict function
   def predict(self, data):
      """
      """

      return self.weights[0] + np.sum(self.weights[1:] * np.array(data))               ##  Sum of ( weights * x-data value) ~ o1*x1 + o2*x2 + o3*x3  =  one value


   def predict_vector(self, data):
      """
      """
      return(np.dot(np.array(data),self.weights[0:7].T))


