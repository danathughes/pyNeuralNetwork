## logisticRegression.py    Dana Hughes    version 1.0     24-November-2014
##
## Logistic Regression model, which predicts a single output value given an 
## input vector.
##
## Revisions:
##   1.0   Initial version, modified from LinearRegressionModel with
##         batch and stochastic gradient descent.
##   1.01  Modified output to be softmax vector
##   1.02  Fixed the predictor.  Converges much better now!
##   1.03  Fixed cost function to have cross-entropy cost

import numpy as np
import random

class LogisticRegressionModel:
   """
   """

   def __init__(self, numVariables, numOutputs):
      """
      Create a new Logistic Regression model with randomized weights
      """

      # Set the weights to zero, including an extra weight as the offset
      self.N = numVariables
      self.M = numOutputs
      self.weights = np.zeros((numVariables + 1, numOutputs))     


   def randomize_weights(self):
      """
      Set all the weights to a value in the range of 1/fan_in
      """

      for i in range(self.N+1):
         for j in range(self.M):
            self.weights[i,j] = (random.random()-0.5)/(self.N)
      


   def sigmoid(self, z):
      """
      """

      return 1.0/(1.0+np.exp(-z))


   def cost(self, data, output):
      """
      Determine the cost (error) of the parameters given the data and labels

      Cost for the softmax is the cross-entropy of the target and predicted 
      output -- cost = -sum_j t_j log(y_j)
      """

      # Add the offset term to the data
      cost = 0.0

      for i in range(len(data)):
         prediction = self.predict(data[i])

         for j in range(self.M):
            cost = cost - output[i][j]*np.log(prediction[j])

      return cost


   def gradient(self, data, output):
      """
      Determine the gradient of the parameters given the data and labels

      Gradient for softmax is dC/dw = dC/dy*dy/dz*dz/dw = (y-t)*x
      """

      gradient = np.zeros((self.N + 1, self.M)) 

      for k in range(len(data)):
         prediction = self.predict(data[k])

         for j in range(self.M):
            gradient[0,j] -= (output[k][j] - prediction[j])

            for i in range(self.N):
               gradient[i+1,j] -= data[k][i]*(output[k][j] - prediction[j])
  
      return gradient/len(data)


   def update_weights(self, dW):
      """
      Update the weights in the model by adding dW
      """

      self.weights += dW


   def predict(self, data):
      """
      Predict the class probabilites given the data
      """

      prediction = np.zeros(self.M)

      for i in range(self.M):         
         prediction[i] = np.exp(self.weights[0,i] + np.sum(self.weights[1:,i]*np.array(data)))

      partition = sum(prediction)

      for i in range(self.M):
         prediction[i] = prediction[i]/partition

      return prediction


   def classify(self, data):
      """
      Classify the data by assigning 1 to the highest probability
      """

      predictions = self.predict(data)
      P_max = np.max(predictions)

      classes = [1 if p == P_max else 0 for p in predictions]
      return classes

      
