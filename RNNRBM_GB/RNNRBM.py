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

      self.initial_rnn = np.zeros((num_rnn, 1))

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

      self.initial_rnn = np.random.uniform(low, high, self.initial_rnn.shape)


   def save_model(self, filename, delimiter = '<DELIMITER>'):
      """
      Write the model parameters to the filename provided
      """

      try:
         outfile = open(filename, 'w')
      except IOError:
         print "Could not open filename '" + filename + "'."
         return

      # Write out the number of units in each layer
      outfile.write("%d%s%d%s%d%s" % (self.num_visible, delimiter, self.num_hidden, delimiter, self.num_rnn, delimiter))

      # Write out the pickle string of each weight matrix
      outfile.write(self.Whv.dumps() + delimiter)
      outfile.write(self.Wuh.dumps() + delimiter)
      outfile.write(self.Wuv.dumps() + delimiter)
      outfile.write(self.Wuu.dumps() + delimiter)
      outfile.write(self.Wvu.dumps() + delimiter)

      outfile.write(self.bias_visible.dumps() + delimiter)
      outfile.write(self.bias_hidden.dumps() + delimiter)
      outfile.write(self.bias_rnn.dumps() + delimiter)

      outfile.write(self.initial_rnn.dumps() + delimiter)

      outfile.close()


   def load_model(self, filename, delimiter = '<DELIMITER>'):
      """
      Load model parameters from the file provided
      """

      try:
         infile = open(filename, 'r')
      except IOError:
         print "Could not open filename '" + filename + "'."
         return

      # Get the number of units in each layer
      data = infile.read()
      data = data.split(delimiter)

      self.num_visible = int(data[0])
      self.num_hidden = int(data[1])
      self.num_rnn = int(data[2])

      # Read each line and convert to weight matrices
      self.Whv = np.loads(data[3])
      self.Wuh = np.loads(data[4])
      self.Wuv = np.loads(data[5])
      self.Wuu = np.loads(data[6])
      self.Wvu = np.loads(data[7])

      self.bias_visible = np.loads(data[8])
      self.bias_hidden = np.loads(data[9])
      self.bias_rnn = np.loads(data[10])

      self.initial_rnn = np.loads(data[11])

      infile.close()


   def cost(self, dataset, output = [], M = 1, k=1):
      """
      Estimate the cost as the RMS error between the dataset and the reconstruction
      Average the reconstruction over N reconstructions
      """

      total_cost = 0.0
      count = 0

      for sequence in dataset:
         
         samples = []
         for i in range(len(sequence)):
            samples.append(np.zeros((self.num_visible, M)))
         
         for i in range(M):
            prior_rnn = self.initial_rnn
            v_guess = np.zeros((self.num_visible, 1))
            for j in range(len(sequence)):
               v_next= self.generate_visible(prior_rnn, v_guess, k)
               v_guess = np.array([sequence[j]]).transpose()
               prior_rnn = self.get_rnn(v_guess, prior_rnn)
               samples[j][:,i] = v_next[:,0]

            
         means = [np.mean(s,1) for s in samples]
         for i in range(len(means)):
            err = np.sum((means[i] - np.array(sequence[i]))**2)
            total_cost = total_cost + err

         count = count + self.num_visible * len(sequence)
 

      return total_cost / count


   def get_weights(self):
      """
      Return the current weights
      """

      return self.Whv, self.Wuh, self.Wuv, self.Wuu, self.Wvu, self.bias_visible, self.bias_hidden, self.bias_rnn, self.initial_rnn 


   def gradient(self, dataset, output=[]):
      """
      Calculate the gradient given the dataset and initial rnn
      """

      N = 0

      grad_Whv = np.zeros(self.Whv.shape)
      grad_Wuh = np.zeros(self.Wuh.shape)
      grad_Wuv = np.zeros(self.Wuv.shape)
      grad_Wuu = np.zeros(self.Wuu.shape)
      grad_Wvu = np.zeros(self.Wvu.shape)
      grad_bv = np.zeros(self.bias_visible.shape)
      grad_bh = np.zeros(self.bias_hidden.shape)
      grad_bu = np.zeros(self.bias_rnn.shape)
      grad_u0 = np.zeros(self.initial_rnn.shape)

      for sequence in dataset:
         N = N + len(sequence)
         dWhv, dWuh, dWuv, dWuu, dWvu, dbv, dbh, dbu, du0 = self.train_sequence(sequence)
         grad_Whv = grad_Whv + dWhv
         grad_Wuh = grad_Wuh + dWuh
         grad_Wuv = grad_Wuv + dWuv
         grad_Wuu = grad_Wuu + dWuu
         grad_Wvu = grad_Wvu + dWvu
         grad_bv = grad_bv + dbv
         grad_bh = grad_bh + dbh     
         grad_bu = grad_bu + dbu
         grad_u0 = grad_u0 + du0

      return grad_Whv/N, grad_Wuh/N, grad_Wuv/N, grad_Wuu/N, grad_Wvu/N, grad_bv/N, grad_bh/N, grad_bu/N, grad_u0/N



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


#   def sample_visible(self, hidden, visible_bias):
#      """
#      Generate a sample of the visible layer given the hidden layer and the dynamic visible bias.
#      """
#
#      P_visible = self.get_probability_visible(hidden, visible_bias)
#
#      v_sample = [1.0 if random.random() < p else 0.0 for p in P_visible]
#      return np.array([v_sample]).transpose()


   def sample_visible(self, hidden, visible_bias):
      """
      Generate a sample of the visible layer from the normal distribution
      """

      # Sample from normal distribution 
      net = np.dot(self.Whv, hidden) + visible_bias
      return np.random.normal(net, 0.1)
#      return net


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


   def generate_rnn_sequence(self, visible_sequence):
      """
      Propagate the values of a sequence of visible units to the rnn units.

      This corresponds to step 1 of the RNN-RBM training algorithm in ICML2012
      """

      rnn_prior = self.initial_rnn

      # The list of RNN units from t=1...T
      rnn = []

      for visible in visible_sequence:
         rnn.append(self.get_rnn(visible, rnn_prior))
         rnn_prior = rnn[-1]

      return rnn


   def generate_bh_sequence(self, rnn_sequence):
      """
      Calculate a sequence of dynamic hidden biasses from the rnn sequence
      """

      bh_sequence = [self.get_bh(self.initial_rnn)]

      # We're only interested in the sequence up to time step T, but could generate
      # The hidden for T+1 if desired
      for i in range(len(rnn_sequence) - 1):
         bh_sequence.append(self.get_bh(rnn_sequence[i]))

      return bh_sequence


   def generate_bv_sequence(self, rnn_sequence):
      """
      Calculate a sequence of dynamic hidden biasses from the rnn sequence
      """

      bv_sequence = [self.get_bv(self.initial_rnn)]

      # We're only interested in the sequence up to time step T, but could generate
      # The hidden for T+1 if desired
      for i in range(len(rnn_sequence) - 1):
         bv_sequence.append(self.get_bv(rnn_sequence[i]))

      return bv_sequence


   def train_sequence(self, visible_sequence, k=1):
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
      rnn_sequence = self.generate_rnn_sequence(visible_sequence)

      # 2a.  Calculate the RBM parameters which depend on the rnn units (bh and bv)
      bh_sequence = self.generate_bh_sequence(rnn_sequence)
      bv_sequence = self.generate_bv_sequence(rnn_sequence)

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
         rnn_prior = self.initial_rnn if i == 0 else rnn_sequence[i-1]
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
         prior_rnn = self.initial_rnn if i == 0 else rnn_sequence[i]
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


   def update_weights(self, dW):
      """
      Add the updates to the corresponding weights
      """

      self.Whv = self.Whv + dW[0]
      self.Wuh = self.Wuh + dW[1]
      self.Wuv = self.Wuv + dW[2]
      self.Wuu = self.Wuu + dW[3]
      self.Wvu = self.Wvu + dW[4]
      self.bias_visible = self.bias_visible + dW[5]
      self.bias_hidden = self.bias_hidden + dW[6]
      self.bias_rnn = self.bias_rnn + dW[7]
      self.initial_rnn = self.initial_rnn + dW[8]
