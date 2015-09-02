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
      self.weights = np.zeros(numVariables + 1)                               ## New class attribute weights(coefficients) # of variables + 1 for the offset No
      for i in range(self.N):
          self.weights[i] = random.random()                                   ## assign random weights
     

   def cost(self, data, output):                                              # data is x axis values?
      """
      """

      # Add the offset term to the data
      cost = 0.0

      for i in range(len(data)):                                              ##
         prediction = self.predict(data[i])                                   ##predict output value for each x-data value
         error = prediction - output[i]                                       ## calculate error for each prediction
         cost = cost + error**2


      return cost/2                                                           ## returns the total "Cost" output from predictions  (Why not divide by average?)


   def gradient(self, data, output):
      """
      Return the gradient of the cost function wrt weights given the data and
      ground truth
      """

      gradient = np.zeros(self.N + 1)

      prediction = self.predict_vector(np.array(data))
      errors = prediction - np.array(output)

      gradient[0] = np.sum(errors)/2

      for j in range(self.N):
         gradient[j+1] = 0
         for i in range(len(errors)):
            gradient[j+1] = gradient[j+1] + data[i][j] * errors[i]
         gradient[j+1] = gradient[j+1]/2

      return gradient


   def train_batch(self, data, output, learning_rate = 0.1, convergence = 0.0001, maxEpochs = 10000):
      """
      """

      epoch = 0

      while self.cost(data, output) > convergence and epoch < maxEpochs:
         print "Epoch", epoch, "- Cost:", self.cost(data,output)
         epoch+=1
         gradient = np.array(self.gradient(data, output))
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
 


   def predict(self, data):                                                              ## Predict function?
      """
      """

      return self.weights[0] + np.sum(self.weights[1:] * np.array(data))               ##  Sum of ( weights * x-data value) ~ o1*x1 + o2*x2 + o3*x3  =  one value


   def predict_vector(self, data):
      """
      """

      return self.weights[0] + np.sum(self.weights[1:] * np.array(data), 1)

      
