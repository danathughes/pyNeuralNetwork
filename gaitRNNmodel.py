from nn.components.layers.RecurrentLayer import *
from nn.components.layers.SigmoidLayer import *
from nn.components.layers.SoftmaxLayer import *
from nn.components.layers.InputLayer import *
from nn.components.connections.FullConnection import *
from nn.components.connections.Bias import *
from nn.components.objectives.MSEObjective import *
from nn.components.objectives.CrossEntropyObjective import *
from nn.models.RecurrentNeuralNetwork import *
from trainers.SGD import SGDTrainer


import numpy as np
import cPickle as pickle

# Build the network
def buildNetwork(rnn_size):
   """
   Build an RNN with softmax output
   """

   net = RecurrentNeuralNetwork()

   input_layer = InputLayer(27)       # Input is 27 unit vector
   recurrent_layer = RecurrentLayer(rnn_size)
   output_layer = SoftmaxLayer(34)

   input_to_recurrent = FullConnection(input_layer.output, recurrent_layer.input)
   recurrent_to_output = FullConnection(recurrent_layer.output, output_layer.input)
   recurrent_bias = Bias(recurrent_layer.input)
   output_bias = Bias(output_layer.input) 

   target_layer = InputLayer(34)      # 34 classes!
   objective_layer = CrossEntropyObjective(output_layer.output, target_layer.output)

   net.addLayer(input_layer)
   net.addLayer(target_layer)
   net.addConnection(input_to_recurrent)
   net.addConnection(recurrent_bias)
   net.addRecurrentLayer(recurrent_layer)
   net.addConnection(recurrent_to_output)
   net.addConnection(output_bias)
   net.addLayer(output_layer)
   net.addObjective(objective_layer)

   net.setInputLayer(input_layer)
   net.setTargetLayer(target_layer)
   net.setOutputLayer(output_layer)
   net.setObjective(objective_layer)

   net.randomize()

   return net


def run():
   # Parameters
   RNN_SIZE = 50
   LEARNING_RATE = 0.00001
   MOMENTUM = 0.9
   WEIGHT_DECAY = 0.0001
   NUMBER_EPOCHS = 10000

   print "Building network and trainer..."
   # Get the model and trainer
   rnn_net = buildNetwork(RNN_SIZE)
   trainer = SGDTrainer(rnn_net, learning_rate = LEARNING_RATE, momentum = MOMENTUM, weight_decay = WEIGHT_DECAY)
   print 

   training_filename = 'training_dataset.pkl'
   validation_filename = 'validation_dataset.pkl'
   test_filename = 'test_dataset.pkl'

   print "Loading training data..."
   f=open(training_filename, 'rb')
   training_dataset = pickle.load(f)
   f.close()

   print "Loading validation data..."
   f=open(validation_filename, 'rb')
   validation_dataset = pickle.load(f)
   f.close()

   print "Loading test data..."
   f=open(test_filename, 'rb')
   test_dataset = pickle.load(f)
   f.close()
 
   print

   print "Training and getting objective!"
   print

   # Let's classify things first and see how well it's starting off as
   training_objective = rnn_net.getSequenceObjective(training_dataset)
   validation_objective = rnn_net.getSequenceObjective(validation_dataset)
   print "Step # 0:\tTraining Objective =", training_objective,", \t Validation Objective =", validation_objective

   objective_file = open('RNN_model_objectives.txt','w')
   objective_file.write('Epoch Number, Training Objective, Validation Objective\n')
   objective_file.write(str(0) + ',' + str(training_objective) + ',' + str(validation_objective) + '\n')

   # Let's start training!
   for i in range(NUMBER_EPOCHS):
      trainer.trainBatch(training_dataset)

      training_objective = rnn_net.getSequenceObjective(training_dataset)
      validation_objective = rnn_net.getSequenceObjective(validation_dataset)
      print "Step #", i+1 ,":\tTraining Objective = ", training_objective,", \t Validation Objective =", validation_objective
      objective_file.write(str(0) + ',' + str(training_objective) + ',' + str(validation_objective) + '\n')
      objective_file.flush()
   
      if (i+1)%25 == 0:
         # Save the model!
         f=open('RNN_model_snapshot_'+str(i+1)+'.pkl','wb')
         pickle.dump(rnn_net, f)
         f.close()

   objective_file.close()

if __name__=='__main__':
  run()
