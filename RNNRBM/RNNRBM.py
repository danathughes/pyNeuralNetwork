import numpy as np
import random
import copy
from Functions.functions import *

class RNNRBM:
   """
   """

   def __init__(self, num_visible, num_hidden, num_rnn):
      """
      """

      self.num_visible = num_visible
      self.num_hidden = num_hidden
      self.num_rnn = num_rnn

      # Weights is a matrix representing the weights between visible units
      # (rows) and hidden unit (columns)

      # Biases are column vectors with the number of hidden or visible units
      self.Whv = np.zeros((num_visible, num_hidden))
      self.Wuh = np.zeros((num_hidden, num_rnn))
      self.Wuv = np.zeros((num_visible, num_rnn))
      self.Wuu = np.zeros((num_rnn, num_rnn))
      self.Wvu = np.zeros((num_rnn, num_visible))

      # Biases
      self.bias_visible = np.zeros((num_visible, 1))
      self.bias_hidden = np.zeros((num_hidden, 1))
      self.bias_rnn = np.zeros((num_rnn, 1))

      # Activation Functions
      self.activate_v = sigmoid
      self.activate_h = sigmoid
      self.activate_u = sigmoid
      self.randomize_weights_and_biases(0.01)


   def randomize_weights_and_biases(self, value_range = 1):
      """
      Set all weights and biases to a value between [-range/2 and range/2]
      """

      low = -value_range/2
      high = value_range/2

      self.Whv = np.random.uniform(low, high, self.Whv.shape)
      self.Wuh = np.random.uniform(low, high, self.Wuh.shape)
      self.Wuv = np.random.uniform(low, high, self.Wuv.shape)
      self.Wuu = np.random.uniform(low, high, self.Wuu.shape)
      self.Wvu = np.random.uniform(low, high, self.Wvu.shape)

      self.bias_visible = np.random.uniform(low, high, self.bias_visible.shape)
      self.bias_hidden = np.random.uniform(low, high, self.bias_hidden.shape)
      self.bias_rnn = np.random.uniform(low, high, self.bias_rnn.shape)


   def get_bv(self, rnn_prior):
      """
      Get the dynamic bias for visible units given the prior rnn layer
      """

      return self.bias_visible + np.dot(self.Wuv, rnn_prior)


   def get_bh(self, rnn_prior):
      """
      Get the dynamic bias for hidden units given the prior rnn layer
      """

      return self.bias_hidden + np.dot(self.Wuh, rnn_prior)


   def get_rnn(self, visible, rnn_prior):
      """
      Get the current rnn units given the prior and visible units
      """

      net = self.bias_rnn + np.dot(self.Wvu, visible) + np.dot(self.Wuu, rnn_prior)
      return self.activate_u(net)


   def get_probability_hidden(self, visible, hidden_bias):
      """
      Returns the probability of setting hidden units to 1, given the 
      visible unit and the dynamic hidden bias.
      """

      return self.activate_h(np.dot(self.Whv.transpose(), visible) + hidden_bias)


   def get_probability_visible(self, hidden, visible_bias):
      """
      Returns the probability of setting visible units to 1, given the
      hidden units and the dynamic visible bias.
      """

      return self.activate_v(np.dot(self.Whv, hidden) + visible_bias)


   def sample_visible(self, hidden, visible_bias):
      """
      Generate a sample of the visible layer given the hidden layer and the dynamic visible bias.
      """

      P_visible = self.get_probability_visible(hidden, visible_bias)

      v_sample = [1.0 if random.random() < p else 0.0 for p in P_visible]
      return np.array([v_sample]).transpose()


   def sample_hidden(self, visible, hidden_bias):
      """
      Generate a sample of the hidden layer given the visible layer and the dynamic hidden bias.
      """

      P_hidden = self.get_probability_hidden(visible, hidden_bias)

      h_sample = [1.0 if random.random() < p else 0.0 for p in P_hidden]
      return np.array([h_sample]).transpose()


   def block_gibbs(self, h0, visible_bias, hidden_bias, k=1):
      """
      Calculate vk and hk after k iterations of Gibbs sampling
      """

      hk = h0

      for i in range(k):
         vk = self.sample_visible(hk, visible_bias)
         hk = self.sample_hidden(vk, hidden_bias)

      return vk, hk


   def generate_visible(self, prior_rnn, v_guess=None, k=5):
      """
      Generate the visible vector at time t given rnn at t-1
      """

      # Calculate the dynamic hidden and visible bias
      bh = self.get_bh(prior_rnn)
      bv = self.get_bv(prior_rnn)

      # Generate v and h using gibbs sampling

      if v_guess == None:
         v_guess = np.zeros((self.num_visible,1))

      h0 = self.sample_hidden(v_guess, bh)
      vk, hk = self.block_gibbs(h0, bv, bh, k)

      return vk


   def generate_rnn_sequence(self, visible_sequence, initial_rnn):
      """
      Propagate the values of a sequence of visible units to the rnn units.

      This corresponds to step 1 of the RNN-RBM training algorithm in ICML2012
      """

      rnn_prior = initial_rnn

      # The list of RNN units from t=1...T
      rnn = []

      for visible in visible_sequence:
         rnn.append(self.get_rnn(visible, rnn_prior))
         rnn_prior = rnn[-1]

      return rnn


   def generate_bh_sequence(self, initial_rnn, rnn_sequence):
      """
      Calculate a sequence of dynamic hidden biasses from the rnn sequence
      """

      bh_sequence = [self.get_bh(initial_rnn)]

      # We're only interested in the sequence up to time step T, but could generate
      # The hidden for T+1 if desired
      for i in range(len(rnn_sequence) - 1):
         bh_sequence.append(self.get_bh(rnn_sequence[i]))

      return bh_sequence


   def generate_bv_sequence(self, initial_rnn, rnn_sequence):
      """
      Calculate a sequence of dynamic hidden biasses from the rnn sequence
      """

      bv_sequence = [self.get_bv(initial_rnn)]

      # We're only interested in the sequence up to time step T, but could generate
      # The hidden for T+1 if desired
      for i in range(len(rnn_sequence) - 1):
         bv_sequence.append(self.get_bv(rnn_sequence[i]))

      return bv_sequence


   def train_sequence(self, visible_sequence, initial_rnn, k=1):
      """
      Perform the training algorithm in ICML2012 on the given sequence

      Returns the gradients in the following manner:
        dC/dWhv - 	the hidden - visible RBM weight matrix
        dC/dWuh - 	the rnn - hidden weight matrix
        dC/dWuv - 	the rnn - visible weight matrix
        dC/dWuu - 	the rnn - rnn weight matrix
        dC/dWvu - 	the visible - rnn weight matrix
        dC/dbv  - 	the visible bias
        dC/dbh  - 	the hidden bias
        dC/dbu  -	the rnn bias
        dC/drnn_init - 	the initial rnn hidden unit
      """
      N = len(visible_sequence)    # How many items in the visible sequence

      # 1.  Propagate to the RNN hidden units using equation (11)
      rnn_sequence = self.generate_rnn_sequence(visible_sequence, initial_rnn)

      # 2a.  Calculate the RBM parameters which depend on the rnn units (bh and bv)
      bh_sequence = self.generate_bh_sequence(initial_rnn, rnn_sequence)
      bv_sequence = self.generate_bv_sequence(initial_rnn, rnn_sequence)

      # 2b.  Sample h_0^T, and perform block Gibbs sampling to get v_n^T and h_n^T
      h0_sequence = []
      hk_sequence = []
      vk_sequence = []

      for i in range(N):
         h0 = self.sample_hidden(visible_sequence[i], bh_sequence[i])
         vk, hk = self.block_gibbs(h0, bv_sequence[i], bh_sequence[i], k)
         h0_sequence.append(h0)
         hk_sequence.append(hk)
         vk_sequence.append(vk)

      # 3.  Use CD-k to estimate the log-likelihood gradient w.r.t. to W, bv^t and bh^t
      grad_LL_W = []
      grad_LL_bv = []
      grad_LL_bh = []

      for i in range(N):
         # What's the 0 and k distributions?
         v0 = visible_sequence[i]
         h0 = h0_sequence[i]
         vk = vk_sequence[i]
         hk = hk_sequence[i]

         # And the bv and bh
         bv = bv_sequence[i]
         bh = bh_sequence[i]

         # And the LL gradients
         grad_LL_W.append(np.dot(v0, h0.transpose()) - np.dot(vk, hk.transpose()))
         grad_LL_bv.append(v0 - vk)
         grad_LL_bh.append(h0 - hk)

      # 4.  Propagate the estimated gradient w.r.t. bv, bh backward through time (BPTT)
      # We're going to have to calculate a whole lotta gradients
      dC_dbv_t = []
      dC_dbh_t = []
 
      dC_dWhv_t = []
      dC_dWuh_t = []
      dC_dWuv_t = []

      for i in range(N):
         bv = bv_sequence[i]
         bh = bh_sequence[i]
         v0 = visible_sequence[i]
         vk = vk_sequence[i]
         h0 = h0_sequence[i]
         hk = hk_sequence[i]

         # Equation 13
         dC_dbv_t.append(vk - v0)

         # Equation 15
         bh_k = self.activate_h(np.dot(self.Whv.transpose(), vk) - bh) 
         bh_0 = self.activate_h(np.dot(self.Whv.transpose(), v0) - bh)
         dC_dbh_t.append(bh_k - bh_0)
         
         # Argument of Equation 14
         Whv_k = np.dot(vk, self.activate_h(np.dot(self.Whv.transpose(), vk) - bh).transpose())
         Whv_0 = np.dot(v0, self.activate_h(np.dot(self.Whv.transpose(), v0) - bh).transpose())
         dC_dWhv_t.append(Whv_k - Whv_0)

         # Argument of Equation 16
         rnn_prior = initial_rnn if i == 0 else rnn_sequence[i-1]
         dC_dWuh_t.append(np.dot(dC_dbh_t[i], rnn_prior.transpose()))

         # Argument of Equation 17
         dC_dWuv_t.append(np.dot(dC_dbv_t[i], rnn_prior.transpose()))
      
         # Equation 18 is simply the sum of Equations 13 and 15, which will be performed after the loop
         # No need to really store this same thing in another list

      # Let's summarize all this stuff before proceeding
      # The cost gradient 
      dC_dWhv = sum(dC_dWhv_t)     # Equation 14
      dC_dWuh = sum(dC_dWuh_t)     # Equation 16
      dC_dWuv = sum(dC_dWuv_t)     # Equation 17
      dC_dbh = sum(dC_dbh_t)       # Equation 18a
      dC_dbv = sum(dC_dbv_t)       # Equation 18b

      # Now we gotta tackle Equation 19
      dC_drnn_t = [None]*N
      
      # Calculate the change in cost w.r.t. rnn layer starting at T and working backward
      dC_drnn_t[N-1] = np.zeros((self.num_rnn, 1))

      for i in range(N-2, -1, -1):
         dC_drnn_t[i] = np.dot(self.Wuu, (dC_drnn_t[i+1]*rnn_sequence[i+1]*(1-rnn_sequence[i+1])))
         dC_drnn_t[i] = dC_drnn_t[i] + np.dot(self.Wuh.transpose(), dC_dbh_t[i+1])
         dC_drnn_t[i] = dC_drnn_t[i] + np.dot(self.Wuv.transpose(), dC_dbv_t[i+1])

      dC_drnn_0 = np.dot(self.Wuu, (dC_drnn_t[0] * rnn_sequence[0] * (1 - rnn_sequence[0])))
      dC_drnn_0 = dC_drnn_0 + np.dot(self.Wuh.transpose(), dC_dbh_t[0])
      dC_drnn_0 = dC_drnn_0 + np.dot(self.Wuv.transpose(), dC_dbv_t[0])

      dC_dbu_t = []
      dC_dWuu_t = []
      dC_dWvu_t = []

      # Arguments for Equations 20 - 22
      for i in range(N):
         dC_dbu_t.append(dC_drnn_t[i] * rnn_sequence[i] * (1 - rnn_sequence[i]))
         prior_rnn = initial_rnn if i == 0 else rnn_sequence[i]
         dC_dWuu_t.append(np.dot(dC_dbu_t[i], prior_rnn.transpose()))
         dC_dWvu_t.append(np.dot(dC_dbu_t[i], visible_sequence[i].transpose()))

      dC_dbu = sum(dC_dbu_t)    # Equation 20
      dC_dWuu = sum(dC_dWuu_t)      # Equation 21
      dC_dWvu = sum(dC_dWvu_t)      # Equation 22

      # All done!  Return the gradients
      # Returns the gradients in the following manner:
      #  dC/dWhv - 	the hidden - visible RBM weight matrix
      #  dC/dWuh - 	the rnn - hidden weight matrix
      #  dC/dWuv - 	the rnn - visible weight matrix
      #  dC/dWuu - 	the rnn - rnn weight matrix
      #  dC/dWvu - 	the visible - rnn weight matrix
      #  dC/dbv  - 	the visible bias
      #  dC/dbh  - 	the hidden bias
      #  dC/dbu  -	the rnn bias
      #  dC/drnn_init - 	the initial rnn hidden unit

      return dC_dWhv, dC_dWuh, dC_dWuv, dC_dWuu, dC_dWvu, dC_dbv, dC_dbh, dC_dbu, dC_drnn_0


   def update_weights(self, dWhv, dWuh, dWuv, dWuu, dWvu, dbv, dbh, dbu):
      """
      Add the updates to the corresponding weights
      """

      self.Whv = self.Whv + dWhv
      self.Wuh = self.Wuh + dWuh
      self.Wuv = self.Wuv + dWuv
      self.Wuu = self.Wuu + dWuu
      self.Wvu = self.Wvu + dWvu
      self.bias_visible = self.bias_visible + dbv
      self.bias_hidden = self.bias_hidden + dbh
      self.bias_rnn = self.bias_rnn + dbu
