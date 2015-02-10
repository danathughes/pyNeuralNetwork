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
##   1.03  Separated training algorithms into a separate file
##         Set up model to accept cost / gradient functions, rather than
##         modifying specific implementations
##         Added cost / gradient functions for sigmoid and softmax layers

import numpy as np
import random

## Cost and gradient functions

def cost_sigmoid(model, dataset, outputs):
   """
   Cost function for sigmoid output units
   """

   # Add the offset term to the data
   cost = 0.0

   for i in range(len(dataset)):
      prediction = model.predict(dataset[i])

      for j in range(model.M):
         cost = cost - ((1.0-outputs[i][j])*np.log(1.0-prediction[j]) + outputs[i][j]*np.log(prediction[j]))

   return cost


def cost_softmax(model, dataset, outputs):
   """
   Cost function for softmax output units (i.e., cross-entropy error
   """

   cost = 0.0

   for i in range(len(dataset)):
      prediction = model.predict(dataset[i])

      for j in range(model.M):
         cost = cost - outputs[i][j]*np.log(prediction[j])

   return cost



def gradient_sigmoid(model, dataset, outputs):
   """
   Gradient for sigmoid output units
   """

   gradient = np.zeros((model.N + 1, model.M)) 

   for k in range(len(dataset)):
      prediction = model.predict(dataset[k])

      for j in range(model.M):
         gradient[0,j] -= prediction[j]*(1.0 - prediction[j])*(outputs[k][j] - prediction[j])

         for i in range(model.N):
            gradient[i+1,j] -= dataset[k][i]*prediction[j]*(1.0 - prediction[j])*(outputs[k][j] - prediction[j])
  
   return gradient/len(dataset)


def gradient_softmax(model, dataset, outputs):
   """
   Gradient for softmax output units
   """

   gradient = np.zeros((model.N + 1, model.M)) 

   for k in range(len(dataset)):
      prediction = model.predict(dataset[k])

      for j in range(model.M):
         gradient[0,j] -= (outputs[k][j] - prediction[j])

         for i in range(model.N):
            gradient[i+1,j] -= dataset[k][i]*(outputs[k][j] - prediction[j])
  
   return gradient/len(dataset)


def predict_sigmoid(model, data):
   """
   Provide a prediction for the data provided
   """

   prediction = np.zeros(model.M)

   for i in range(model.M):         
      prediction[i] = model.sigmoid(model.weights[0,i] + np.sum(model.weights[1:,i]*np.array(data)))

   return prediction


def predict_softmax(model, data):
   """
   Provide a prediction for the data provided
   """

   prediction = np.zeros(model.M)

   for i in range(model.M):         
      prediction[i] = np.exp(model.weights[0,i] + np.sum(model.weights[1:,i]*np.array(data)))

   partition = sum(prediction)

   for i in range(model.M):
      prediction[i] = prediction[i]/partition

   return prediction

# Create constant tuples of cost / gradient pairs for use with the Logistic 
# Regression models

SIGMOID = (cost_sigmoid, gradient_sigmoid, predict_sigmoid)
SOFTMAX = (cost_softmax, gradient_softmax, predict_softmax)


class LogisticRegressionModel:
   """
   """

   def __init__(self, numVariables, numOutputs, activation = SOFTMAX):
      """
      Create a new Logistic Regression model with randomized weights
      """

      # Set the weights to zero, including an extra weight as the offset
      self.N = numVariables
      self.M = numOutputs
      self.weights = np.zeros((numVariables + 1, numOutputs))     
      self.activation_cost = activation[0]
      self.activation_gradient = activation[1]
      self.activation_predict = activation[2]


   def randomize_weights(self):
      """
      Set all the weights to a value in the range of 1/fan_in
      """

      for i in range(self.N+1):
         for j in range(self.M):
            self.weights[i,j] = (random.random()-0.5)/(2.0*self.N)


   def sigmoid(self, z):
      """
      """

      return 1.0/(1.0+np.exp(-z))


   def cost(self, dataset, outputs):
      """
      Determine the cost (error) of the parameters given the data and labels
      """

      return self.activation_cost(self, dataset, outputs)


   def gradient(self, dataset, outputs):
      """
      Determine the gradient of the parameters given the data and labels

      Gradient for the sigmoid output is dy/dx = x*y*(1-y)
      """

      return self.activation_gradient(self, dataset, outputs)


   def update_weights(self, dW):
      """
      Update the weights in the model by adding dW
      """

      self.weights += dW


   def predict(self, data):
      """
      Predict the class probabilites given the data
      """

      return self.activation_predict(self, data)


   def classify(self, data):
      """
      Classify the data by assigning 1 to the highest prediction probability
      """
      
      predictions = self.predict(data)
      P_max = np.max(predictions)

      classes = [1 if p == P_max else 0 for p in predictions]
      return classes
      
