import numpy as np
import random
import copy

class RTRBM:
   """
   """

   def __init__(self, num_visible, num_hidden):
      """
      """

      self.num_visible = num_visible
      self.num_hidden = num_hidden

      # Weights is a matrix representing the weights between visible units
      # (rows) and hidden unit (columns)

      # Biases are column vectors with the number of hidden or visible units
      self.weights = np.zeros((num_visible, num_hidden))
      self.weights_HH = np.zeros((num_hidden, num_hidden))
      self.weights_VH = np.zeros((num_visible, num_hidden))
      self.bias_visible = np.zeros((num_visible, 1))
      self.bias_hidden = np.zeros((num_hidden, 1))
      self.h_0 = [0.5,0.5]
      self.sigma_vis = 0.15

      self.randomize_weights_and_biases(0.1)


   def randomize_weights_and_biases(self, value_range = 1):
      """
      Set all weights and biases to a value between [-range/2 and range/2]
      """

      for i in range(self.num_visible):
         for j in range(self.num_hidden):
            self.weights[i,j] = value_range*random.random() - value_range/2
            self.weights_VH[i,j] = value_range*random.random() - value_range/2

      for i in range(self.num_visible):
         self.bias_visible[i,0] = value_range*random.random() - value_range/2

      for i in range(self.num_hidden):
         self.bias_hidden[i,0] = value_range*random.random() - value_range/2
#         self.h_0[i,0] = value_range*random.random() - value_range/2

      for i in range(self.num_hidden):
         for j in range(self.num_hidden):
            self.weights_HH[i,j] = value_range*random.random() - value_range/2


   def sigmoid(self, z):
      """
      """

      return 1.0 / (1.0 + np.exp(-z))


   def get_bh(self, prior_hidden):
      """
      """

      return self.bias_hidden + np.dot(self.weights_HH, prior_hidden)


   def get_bv(self, prior_hidden):
      """
      """

      return self.bias_visible + np.dot(self.weights_VH, prior_hidden)


   def get_probability_hidden(self, visible, prior_hidden):
      """
      Returns the probability of setting hidden units to 1, given the 
      visible unit.
      """

      # h = sigmoid(W'v + c)
      bh = self.get_bh(prior_hidden)
      return self.sigmoid(np.dot(self.weights.transpose(), visible) + bh)


   def get_probability_visible(self, hidden, prior_hidden):
      """
      Returns the probability of setting visible units to 1, given the
      hidden units.
      """

      bv = self.get_bv(prior_hidden)
      return self.sigmoid(np.dot(self.weights, hidden) + bv)


   def sample_visible(self, hidden, prior_hidden):
      """
      Generate a sample of the visible layer given the hidden layer.
      """

      return np.random.normal(self.get_probability_visible(hidden, prior_hidden), self.sigma_vis)


   def sample_hidden(self, visible, prior_hidden):
      """
      Generate a sample of the hidden layer given the visible layer.
      """

      P_hidden = self.get_probability_hidden(visible, prior_hidden)

      h_sample = [1.0 if random.random() < p else 0.0 for p in P_hidden]
      return np.array([h_sample]).transpose()


   def contrastive_divergence(self, v0, prior_hidden, k=1):
      """
      Perform CD-k for the given data point
      """

      # Calculate an h0 given the v0
      h0 = self.sample_hidden(v0, prior_hidden)
      p_h0 = self.get_probability_hidden(v0, prior_hidden)

      # We'll need to iteratively sample to get the next values.  We'll start
      # with k=0 and iterate
      vk = v0
      hk = h0

      # Now calculate vk and hk
      for i in range(k):
         vk = self.sample_visible(hk, prior_hidden)
         hk = self.sample_hidden(vk, prior_hidden)
      p_hk = self.get_probability_hidden(v0, prior_hidden)     

      # Compute positive and negative as the outer product of these
      positive = np.dot(v0, p_h0.transpose())
      negative = np.dot(vk, p_hk.transpose())

      # Calculate the delta-weight and delta-biases
      delta_weights = positive - negative
      delta_visible_bias = v0 - vk
      delta_hidden_bias = p_h0 - p_hk

      # Calcualte the delta-weights for the W_hh and W_vh matrices
      delta_weights_HH = np.dot(delta_hidden_bias, prior_hidden.transpose())
      delta_weights_VH = np.dot(delta_visible_bias, prior_hidden.transpose())

      # Return these--let the learning rule handle them
      return delta_weights, delta_weights_HH, delta_weights_VH, delta_visible_bias, delta_hidden_bias


   def train_sequence(self, sequence, h_init = None, k = 1):
      """
      """

      if h_init==None:
         h_init = self.h_0

      DW = np.zeros(self.weights.shape)
      DW_HH = np.zeros(self.weights_HH.shape)
      DW_VH = np.zeros(self.weights_VH.shape)
      DB_vis = np.zeros(self.bias_visible.shape)
      DB_hid = np.zeros(self.bias_hidden.shape)
      
      h_prior = np.array([h_init]).transpose()

      for data in sequence:
         # Grab the current visible datapoint and previous hidden state, and 
         # Perform CD
         vis = np.array([data]).transpose()
         dW, dW_HH, dW_VH, dB_vis, dB_hid = self.contrastive_divergence(vis, h_prior, k)
         
         # Update the weights
         DW = DW + dW
         DW_HH = DW_HH + dW_HH
         DW_VH = DW_VH + dW_VH
         DB_vis = DB_vis + dB_vis
         DB_hid = DB_hid + dB_hid

         # Set h_prior to the probability of h given current vis and h_prior
         h_prior = self.get_probability_hidden(vis, h_prior)

      DW = DW / len(sequence)
      DW_HH = DW_HH / len(sequence)
      DW_VH = DW_VH / len(sequence)
      DB_vis = DB_vis / len(sequence)
      DB_hid = DB_hid / len(sequence)

      return DW, DW_HH, DW_VH, DB_vis, DB_hid


   def train_epoch(self, dataset, h_init = None, learning_rate = 0.001, k = 1):
      """
      """

      err = 0.0

      for sequence in dataset:
         dW, dW_HH, dW_VH, dB_vis, dB_hid = self.train_sequence(sequence, h_init, k)
         self.weights = self.weights + dW / len(dataset)
         self.weights_HH = self.weights_HH + dW_HH / len(dataset)
         self.weights_VH = self.weights_VH + dW_VH / len(dataset)
         self.bias_visible = self.bias_visible + dB_vis / len(dataset)
         self.bias_hidden = self.bias_hidden + dB_hid / len(dataset)

      return err/len(dataset)
