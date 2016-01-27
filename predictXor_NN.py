import random
import matplotlib.pyplot as plt

from nn.models.NeuralNetwork import NeuralNetwork
from nn.components.connections.FullConnection import FullConnection
from nn.components.connections.Bias import Bias
from nn.components.layers.InputLayer import InputLayer
from nn.components.layers.SigmoidLayer import SigmoidLayer
from nn.components.layers.TanhLayer import TanhLayer
from nn.components.layers.SoftmaxLayer import SoftmaxLayer
from nn.components.objectives.MSEObjective import MSEObjective
from nn.components.objectives.CrossEntropyObjective import CrossEntropyObjective
from trainers.SGD import SGDTrainer
from trainers.PSO import PSOTrainer

import numpy as np

if __name__ == '__main__':
#   training_set_X, training_set_Y = load_isolet('/home/dana/Desktop/Research/deepLearning/datasets/data/isolet_train.txt')
#   test_set_X, test_set_Y = load_isolet('/home/dana/Desktop/Research/deepLearning/datasets/data/isolet_test.txt')
   training_set_X = np.array([[0,0], [0,1], [1,0], [1,1]])
   training_set_Y = np.array([[0,1], [1,0], [1,0], [0,1]])

   test_set_X = [[0,0], [0,1], [1,0], [1,1]]
   test_set_Y = [[0], [1], [1], [0]]
   # How many variables?
   numVariables = len(training_set_X[0])
   numHidden = 2
   numClasses = 2

   training_set = (training_set_X, training_set_Y)



   # Create the model
   print "Creating model...",
   input_layer = InputLayer(numVariables)
   hidden_layer = TanhLayer(numHidden)
   output_layer = SoftmaxLayer(numClasses)
   target_layer = InputLayer(numClasses)
   input_to_hidden = FullConnection(input_layer.output, hidden_layer.input)
   hidden_to_output = FullConnection(hidden_layer.output, output_layer.input)
   hidden_bias = Bias(hidden_layer.input)
   output_bias = Bias(output_layer.input)
#   objective = MSEObjective(output_layer.output, target_layer.output)
   objective = CrossEntropyObjective(output_layer.output, target_layer.output)

   net = NeuralNetwork()
   net.addLayer(input_layer)
   net.addLayer(target_layer)
   net.addConnection(input_to_hidden)
   net.addConnection(hidden_bias)
   net.addLayer(hidden_layer)
   net.addConnection(hidden_to_output)
   net.addConnection(output_bias)
   net.addLayer(output_layer)
   net.addObjective(objective)
   net.setInputLayer(input_layer)
   net.setTargetLayer(target_layer)
   net.setOutputLayer(output_layer)
   net.setObjective(objective)

   net.randomize()

   print "Done"

   print "Creating a trainer..."
   trainer = SGDTrainer(net, learning_rate=0.1, momentum=0.0, weight_decay=0.00001)
#   trainer = PSOTrainer(net, number_particles=20, initial_weight_range=(-5.0,5.0))
   print "Done"

   print "Training..."
   for i in range(1000000):
     trainer.trainBatch(training_set)
#     print "Iteration", i, "\tObjetive =", trainer.global_best
     print "Iteration", i, "\tObjetive =", net.getObjective()
   print "Results:"
   print net.getOutput()

