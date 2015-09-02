import numpy as np
import random
import copy

class TRBM:
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
      self.bias_visible = np.zeros((num_visible, 1))
      self.bias_hidden = np.zeros((num_hidden, 1))

 
      # These are from Hinton's paper
      self.A = []
      self.C = []

      for i in range(self.num_time_delay):
         self.A.append(np.zeros((self.num_visible, self.num_visible)))
         self.C.append(np.zeros((self.num_visible, self.num_hidden)))

      self.randomize_weights_and_biases(0.01)


   def free_energy(self, data):
      """
      Calculate the free energy formula for a single datapoint
      """

      v0 = np.array([data]).transpose()

      wx_b = np.dot(self.weights.transpose(), v0) + self.bias_hidden
      vbias_term = np.dot(v0.transpose(), self.bias_visible)[0,0]
      hidden_term = np.sum(np.log(1.0 + np.exp(wx_b)))

      return - hidden_term - vbias_term


   def pseudolikelihood(self, dataset):
      """
      Calculate the pseudolikelihood by stochasically approximating the 
      log probability of each bit
      """

      PL = 0

      num_bits = self.num_visible

      for data in dataset:
         E_xi = self.free_energy(data)
         bit_flip_num = random.randrange(0,num_bits)
         data_flip = copy.copy(data)
         data_flip[bit_flip_num] = 1 - data_flip[bit_flip_num]
         E_xi_flip = self.free_energy(data_flip)
         PL = PL + num_bits*np.log(self.sigmoid(E_xi_flip - E_xi))

      return PL
         

   def likelihood(self, dataset):
      """
      Calculate the likelihood by calculating the log probability of each bit
      """

      PL = 0

      num_bits = self.num_visible

      for data in dataset:
         for i in range(num_bits):
            E_xi = self.free_energy(data)
            data_flip = copy.copy(data)
            data_flip[i] = 1 - data_flip[i]
            E_xi_flip = self.free_energy(data_flip)
            PL = PL + np.log(self.sigmoid(E_xi_flip - E_xi))

      return PL


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

      for i in range(self.num_time_delay):
         for j in range(self.num_visible):
            for k in range(self.num_visible):
               self.A[i][j,k] = value_range*random.random() - value_range/2
            for k in range(self.num_hidden):
               self.C[i][j,k] = value_range*random.random() - value_range/2



   def sigmoid(self, z):
      """
      """

      return 1.0 / (1.0 + np.exp(-z))


   def get_BV(self, visibles):
      """
      Return the dynamic bias given visibles V_{t-1}...V_{t-n}
      """

      BV = np.zeros((self.num_visible, 1))

      for i in range(self.num_time_delay):
         BV = BV + np.dot(self.A[i].transpose(), visibles[i])

      return BV


   def get_BH(self, visibles):
      """
      Return the dynamic vias for hidden units given visibles V_{t-1}...V_{t-n}
      """

      BH = np.zeros((self.num_hidden, 1))

      for i in range(self.num_time_delay):
         BH = BH + np.dot(self.C[i].transpose(), visibles[i])

      return BH


   def get_probability_hidden(self, visible, visibles):
      """
      Returns the probability of setting hidden units to 1, given the 
      visible unit.
      """

      # h = sigmoid(W'v + c)
      BH = self.get_BH(visibles)

      return self.sigmoid(np.dot(self.weights.transpose(), visible) + self.bias_hidden + BH)


   def get_probability_visible(self, hidden, visibles):
      """
      Returns the probability of setting visible units to 1, given the
      hidden units.
      """

      BV = self.get_BV(visibles)

      return self.sigmoid(np.dot(self.weights, hidden) + self.bias_visible + BV)


   def sample_visible(self, hidden, visibles):
      """
      Generate a sample of the visible layer given the hidden layer.
      """

      P_visible = self.get_probability_visible(hidden, visibles)

      v_sample = [1.0 if random.random() < p else 0.0 for p in P_visible]
      return np.array([v_sample]).transpose()


   def sample_hidden(self, visible, visibles):
      """
      Generate a sample of the hidden layer given the visible layer.
      """

      P_hidden = self.get_probability_hidden(visible, visibles)

      h_sample = [1.0 if random.random() < p else 0.0 for p in P_hidden]
      return np.array([h_sample]).transpose()


   def contrastive_divergence(self, v0, visibles, k=1):
      """
      Perform CD-k for the given data point
      """

      # Calculate an h0 given the v0
      h0 = self.sample_hidden(v0, visibles)

      # We'll need to iteratively sample to get the next values.  We'll start
      # with k=0 and iterate
      vk = v0
      hk = h0

      # Now calculate vk and hk
      for i in range(k):
         vk = self.sample_visible(hk, visibles)
         hk = self.sample_hidden(vk, visibles)

      # Compute positive and negative as the outer product of these
      positive = np.dot(v0, h0.transpose())
      negative = np.dot(vk, hk.transpose())

      # Calculate the delta-weight and delta-biases
      delta_weights = positive - negative
      delta_visible_bias = v0 - vk
      delta_hidden_bias = h0 - hk

      delta_A = []
      delta_C = []

      for i in range(self.num_time_delay):
         A_pos = np.dot(visibles[i], v0.transpose())
         A_neg = np.dot(visibles[i], vk.transpose())
         delta_A.append(A_pos - A_neg)
         C_pos = np.dot(visibles[i], h0.transpose())
         C_neg = np.dot(visibles[i], hk.transpose())
         delta_C.append(C_pos - C_neg)

      # Return these--let the learning rule handle them
      return delta_weights, delta_visible_bias, delta_hidden_bias, delta_A, delta_C


   def train_epoch(self, sequence, learning_rate = 0.1, k = 1):
      """
      """

      for n in range(self.num_time_delay, len(sequence)):
         v0 = np.array([sequence[n]]).transpose()
         visibles = []
         for i in range(self.num_time_delay):
            visibles.append(np.array([sequence[n-self.num_time_delay+i]]).transpose())

         dW, db_vis, db_hid, dA, dC = self.contrastive_divergence(v0, visibles, k)

         self.weights = self.weights + learning_rate*dW / len(sequence)
         self.bias_hidden = self.bias_hidden + learning_rate*db_hid / len(sequence)
         self.bias_visible = self.bias_visible + learning_rate*db_vis / len(sequence)

         for i in range(self.num_time_delay):
            self.A[i] = self.A[i] + learning_rate*dA[i] / len(sequence)
            self.C[i] = self.C[i] + learning_rate*dC[i] / len(sequence)

