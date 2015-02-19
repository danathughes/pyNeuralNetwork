import numpy as np
import random
import copy

from Functions.functions import sigmoid

class TRBM:
   """
   """

   def __init__(self, num_visible, num_hidden, num_delays):
      """
      A Temporal RBM
      """

      # Do we add one to num_delays?

      self.num_visible = num_visible
      self.num_hidden = num_hidden
      self.num_delays = num_delays

      # Weights are matrices representing the weights between visible units
      # (rows) and hidden units (columns)

      # Biases are column vectors with the number of hidden or visible units
      # There is a list of weights and biases, where the index represents 
      # the time delay, i.e., weights[1] = W_{t-1}, etc.


      self.weights = np.zeros((num_visible, num_hidden))
      self.A = []
      self.B = []
      self.C = []
      self.bias_visible = np.zeros((num_visible, 1))
      self.bias_hidden = np.zeros((num_hidden, 1))

      for i in range(self.num_delays):
         self.C.append(np.zeros((num_visible, num_hidden)))
         self.A.append(np.zeros((num_visible, num_visible)))
         self.B.append(np.zeros((num_hidden, num_hidden)))

      self.randomize_weights_and_biases(8*np.sqrt(6.0/(self.num_hidden + self.num_visible)))


   def randomize_weights_and_biases(self, value_range = 1):
      """
      Set all weights and biases to a value between [-range/2 and range/2]
      """

      self.weights = np.array(np.random.uniform(-value_range/2, value_range/2, self.weights.shape))

      self.bias_visible = np.array(np.random.uniform(-value_range/2, value_range/2, self.bias_visible.shape))

      self.bias_hidden = np.array(np.random.uniform(-value_range/2, value_range/2, self.bias_hidden.shape))

      for n in range(self.num_delays):
         self.C[n] = np.array(np.random.uniform(-value_range/2, value_range/2, self.C[n].shape))
         self.B[n] = np.array(np.random.uniform(-value_range/2, value_range/2, self.B[n].shape))
         self.A[n] = np.array(np.random.uniform(-value_range/2, value_range/2, self.A[n].shape))



   def get_bias_hidden(self, visibles, hiddens):
      """
      Returns B_H, given the prior visible and hidden terms

      Assume visibles[0] = V{t-1}, visibles[1] = V{t-2}, etc.
      Similar for hiddens
      """

      B_H = np.zeros((self.num_hidden,1)) + self.bias_hidden

      for n in range(self.num_delays):
         B_H = B_H + np.dot(self.B[n].transpose(), hiddens[n])
         B_H = B_H + np.dot(self.C[n].transpose(), visibles[n])

      return B_H


   def get_bias_visible(self, visibles):
      """
      Returns B_V, given the prior visible terms

      Assume visibles[0] = V{t-1}, visibles[1] = V{t-2}, etc...
      """

      B_V = np.zeros((self.num_visible,1)) + self.bias_visible

      for n in range(self.num_delays):
         B_V = B_V + np.dot(self.A[n].transpose(), visibles[n])

      return B_V


   def get_probability_hidden(self, visibles, hiddens):
      """
      Returns the probability of setting hidden units to 1, given the 
      history of hidden and visible units.

      Assumes visibles[0] = V{t}, visibles[1] = V{t-1}, ...
      and hiddens[0] = H{t-1}, hiddens[1] = H{t-2}, ...

      visibles should have one extra entry than hiddens
      """

      # H = sigmoid(W'V_t + B_H(V_{t-m...t-1}, H{t-m...t-1}))
      B_H = self.get_bias_hidden(visibles[1:], hiddens)
      return sigmoid(np.dot(self.weights.transpose(), visibles[0]) + B_H)


   def get_probability_visible(self, visibles, hidden):
      """
      Returns the probability of setting visible units to 1, given the
      the hidden units and history of visible units.

      Assume visibles[0] = V{t-1}, visibles[1] = V{t-2}, etc.
      Then hiddens = H{t}
      """

      B_V = self.get_bias_visible(visibles)
      return sigmoid(np.dot(self.weights, hidden) + B_V)


   def sample_visible(self, visibles, hiddens):
      """
      Generate a sample of the visible layer given the hidden layer.
      """

      P_visible = self.get_probability_visible(visibles, hiddens)

      v_sample = [1.0 if random.random() < p else 0.0 for p in P_visible]
      return np.array([v_sample]).transpose()


   def sample_hidden(self, visibles, hiddens):
      """
      Generate a sample of the hidden layer given the visible layer.
      """

      P_hidden = self.get_probability_hidden(visibles, hiddens)

      h_sample = [1.0 if random.random() < p else 0.0 for p in P_hidden]
      return np.array([h_sample]).transpose()


   def contrastive_divergence(self, visibles, k=1):
      """
      Perform CD-k for the given subsequence
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
         self.bias_visibile = self.bias_visible + learning_rate*db_vis

