import numpy as np
import random
import copy

class CRBM:
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
      self.bias_visible = np.zeros((num_visible, 1))
      self.bias_hidden = np.zeros((num_hidden, 1))

      # Standard deviation on the noise 
      self.sigma_visible = np.zeros((num_visible, 1))
      self.sigma_hidden = np.zeros((num_hidden, 1))

      self.a_visible = np.zeros((num_visible, 1))
      self.a_hidden = np.zeros((num_hidden, 1))

      self.sigma_visible[:,:] = 0.2
      self.sigma_hidden[:,:] = 0.2
      self.a_visible[:,:] = 0.1
      self.a_hidden[:,:] = 0.1

      # Are these needed?
      self.lo = 0.0
      self.hi = 1.0

      self.randomize_weights_and_biases()


   def randomize_weights_and_biases(self, value_range = 0.01):
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



   def free_energy(self, data):
      """
      Calculate the free energy formula for a single datapoint
      """



   def pseudolikelihood(self, dataset):
      """
      Calculate the pseudolikelihood by stochasically approximating the 
      log probability of each bit
      """

         

   def likelihood(self, dataset):
      """
      Calculate the likelihood by calculating the log probability of each bit
      """


   #==== Should go in functions ====
   def sigmoid(self, z, a = 1):
      """
      """

      return 1.0 / (1.0 + np.exp(-a*z))


   def phi(self, x, a=1.0, lo=0.0, hi=1.0):
      """
      """

      return self.lo + (self.hi - self.lo)*(1.0/(1.0 + np.exp(-a*x))
   #===== END OF ALL THAT!!! ====

#   def get_probability_hidden(self, visible):
#      """
#      Returns the probability of setting hidden units to 1, given the 
#      visible unit.
#      """
#
#      # h = sigmoid(W'v + c)
#      return self.sigmoid(np.dot(self.weights.transpose(), visible) + self.bias_hidden)
#
#
#   def get_probability_visible(self, hidden):
#      """
#      Returns the probability of setting visible units to 1, given the
#      hidden units.
#      """
#
#      return self.sigmoid(np.dot(self.weights, hidden) + self.bias_visible)


   def sample_visible(self, hidden):
      """
      Generate a sample of the visible layer given the hidden layer.
      """

#      P_visible = self.get_probability_visible(hidden)

#      v_sample = [1.0 if random.random() < p else 0.0 for p in P_visible]
#      return np.array([v_sample]).transpose()

      np.dot(self.weights, hidden) + 


   def sample_hidden(self, visible):
      """
      Generate a sample of the hidden layer given the visible layer.
      """

#      P_hidden = self.get_probability_hidden(visible)

#      h_sample = [1.0 if random.random() < p else 0.0 for p in P_hidden]
#      return np.array([h_sample]).transpose()

      np.dot(self.weights.transpose(), visible)


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

