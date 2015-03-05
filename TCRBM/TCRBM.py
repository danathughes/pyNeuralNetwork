import numpy as np
import random
import copy

class CRBM:
   """
   """

   def __init__(self, num_visible, num_hidden, num_time_delay):
      """
      """

      self.num_visible = num_visible
      self.num_hidden = num_hidden
      self.num_time_delay = num_time_delay

      # Weights is a matrix representing the weights between visible units
      # (rows) and hidden unit (columns)

      # Biases are column vectors with the number of hidden or visible units
      self.weights = np.zeros((num_visible, num_hidden))
      self.autoregressive_weights = []
      self.visible_hidden_weights = []
      self.bias_visible = np.zeros((num_visible, 1))
      self.bias_hidden = np.zeros((num_hidden, 1))
      self.A_visible = np.ones((num_visible, 1))
      self.A_hidden = np.ones((num_hidden, 1))
      self.sigma = 0.2
      self.lo = 0.0
      self.hi = 1.0

      for i in range(self.num_time_delays):
         self.autoregressive_weights.append(np.zeros((num_visible, num_visible))
         self.visible_hidden_weights.append(np.zeros((num_hidden, num_visible))

      self.randomize_weights_and_biases(0.1)


   def randomize_weights_and_biases(self, value_range = 1):
      """
      Set all weights and biases to a value between [-range/2 and range/2]
      """

      for i in range(self.num_visible):
         for j in range(self.num_hidden):
            self.weights[i,j] = value_range*random.random() - value_range/2

      for i in range(self.num_visible):
         self.bias_visible[i,0] = value_range*random.random() - value_range/2

      for i in range(self.num_hidden):
         self.bias_hidden[i,0] = value_range*random.random() - value_range/2


   def sigmoid(self, z, A):
      """
      """

      return self.lo + (self.hi-self.lo) / (1.0 + np.exp(-A*z))


   def sample_visible(self, hidden):
      """
      Generate a sample of the visible layer given the hidden layer.
      """

      v = np.dot(self.weights, hidden) + self.bias_visible
      v += np.random.normal(0,self.sigma, (self.num_visible, 1))
      return self.sigmoid(v, self.A_visible)


   def sample_hidden(self, visible):
      """
      Generate a sample of the hidden layer given the visible layer.
      """

      h = np.dot(self.weights.transpose(), visible) + self.bias_hidden
      h += np.random.normal(0,self.sigma, (self.num_hidden, 1))
      return self.sigmoid(h, self.A_hidden)


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
      delta_A_hidden = h0*h0 - hk*hk
      delta_A_visible = v0*v0 - vk*vk

      # Return these--let the learning rule handle them
      return delta_weights, delta_visible_bias, delta_hidden_bias, delta_A_visible, delta_A_hidden


   def train_epoch(self, dataset, learning_rate = 0.5, k = 1):
      """
      """

      total_err = 0.0

      dW = np.zeros(self.weights.shape)
      dB_vis = np.zeros(self.bias_visible.shape)
      dB_hid = np.zeros(self.bias_hidden.shape)
      dA_vis = np.zeros(self.A_visible.shape)
      dA_hid = np.zeros(self.A_hidden.shape)

      for data in dataset:
         dw, db_vis, db_hid, da_vis, da_hid = self.contrastive_divergence(np.array([data]).transpose(), k)
         dW = dW + dw
         dB_vis = dB_vis + db_vis
         dB_hid = dB_hid + db_hid
         dA_vis = dA_vis + da_vis
         dA_hid = dA_hid + da_hid

         total_err += np.sum(db_vis*db_vis)

      dW = dW / len(dataset)
      dB_vis = dB_vis / len(dataset)
      dB_hid = dB_hid / len(dataset)
      dA_vis = dA_vis / len(dataset)
      dA_hid = dA_hid / len(dataset)

      self.weights = self.weights + learning_rate*dW
      self.bias_hidden = self.bias_hidden + learning_rate*dB_hid
      self.bias_visible = self.bias_visible + learning_rate*dB_vis
      self.A_hidden = self.A_hidden + learning_rate*dA_hid/(self.A_hidden*self.A_hidden)
      self.A_visible = self.A_visible + learning_rate*dA_vis/(self.A_visible*self.A_visible)

      return total_err / len(dataset)
