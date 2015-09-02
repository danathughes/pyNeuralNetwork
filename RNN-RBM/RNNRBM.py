import numpy as np
import random
import copy

from Functions.functions import *

class RNNRBM:
   """
   """

   def __init__(self, num_visible, num_hidden, num_recurrent):
      """
      """

      self.num_visible = num_visible
      self.num_hidden = num_hidden
      self.num_recurrent = num_recurrent

      # Weights is a matrix representing the weights between visible units
      # (rows) and hidden unit (columns)

      # Biases are column vectors with the number of hidden or visible units
      self.Whv = np.zeros((num_visible, num_hidden))
      self.Wuh = np.zeros((num_hidden, num_recurrent))
      self.Wuv = np.zeros((num_visible, num_recurrent))
      self.Wuu = np.zeros((num_recurrent, num_recurrent))
      self.Wvu = np.zeros((num_recurrent, num_visible))
      self.bv = np.zeros((num_visible, 1))
      self.bh = np.zeros((num_hidden, 1))
      self.bu = np.zeros((num_recurrent, 1))

      self.randomize_weights_and_biases(0.01)


   def randomize_weights_and_biases(self, value_range = 1):
      """
      Set all weights and biases to a value between [-range/2 and range/2]
      """

      self.Whv = np.random.uniform(-value_range/2, value_range/2, self.Whv.shape)
      self.Wuh = np.random.uniform(-value_range/2, value_range/2, self.Wuh.shape)
      self.Wuv = np.random.uniform(-value_range/2, value_range/2, self.Wuv.shape)
      self.Wuu = np.random.uniform(-value_range/2, value_range/2, self.Wuu.shape)
      self.bv = np.random.uniform(-value_range/2, value_range/2, self.bv.shape)
      self.bh = np.random.uniform(-value_range/2, value_range/2, self.bh.shape)
      self.bu = np.random.uniform(-value_range/2, value_range/2, self.bu.shape)


   def get_probability_hidden(self, visible):
      """
      Returns the probability of setting hidden units to 1, given the 
      visible unit.
      """


   def get_probability_visible(self, hidden):
      """
      Returns the probability of setting visible units to 1, given the
      hidden units.
      """


   def sample_visible(self, hidden):
      """
      Generate a sample of the visible layer given the hidden layer.
      """


   def sample_hidden(self, visible):
      """
      Generate a sample of the hidden layer given the visible layer.
      """


   def sample_recurrent_layer(self, visible, recurrent):
      """
      Get the value of the recurrent layer given the visible and previous recurrent layer
      """

      return tanh(self.bu + np.dot(self.Wuu, recurrent) + np.dot(self.Wvu, visible))


   def get_bv(self, recurrent):
      """
      Get the dynamic visible bias
      """

      return self.bv + np.dot(self.Wuv, recurrent)


   def get_bh(self, recurrent):
      """
      Get the dynamic hidden bias
      """

      return self.bh + np.dot(self.Wuh, recurrent)


   def contrastive_divergence(self, v0, k=1):
      """
      Perform CD-k for the given data point
      """

      # Calculate an h0 given the v0
      h0 = self.sample_hidden(v0)

      # We'll need to iteratively sample to get the next values.  We'll start
      # with k=0 and iterate
      vk = v0
      hk = h0

      # Now calculate vk and hk
      for i in range(k):
         vk = self.sample_visible(hk)
         hk = self.sample_hidden(vk)

      # Compute positive and negative as the outer product of these
      positive = np.dot(v0, h0.transpose())
      negative = np.dot(vk, hk.transpose())

      # Calculate the delta-weight and delta-biases
      delta_weights = positive - negative
      delta_visible_bias = v0 - vk
      delta_hidden_bias = h0 - hk

      # Return these--let the learning rule handle them
      return delta_weights, delta_visible_bias, delta_hidden_bias


   def train_epoch(self, dataset, learning_rate = 0.001, k = 1):
      """
      """

      for data in dataset:
         dW, db_vis, db_hid = self.contrastive_divergence(np.array([data]).transpose(), k)

         self.weights = self.weights + learning_rate*dW
         self.bias_hidden = self.bias_hidden + learning_rate*db_hid
         self.bias_visible = self.bias_visible + learning_rate*db_vis

