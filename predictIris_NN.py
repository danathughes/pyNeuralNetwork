## predictIris.py
##
## Simple script to predict iris using neural network code
##


import random
from datasets.iris import *
import Preprocess.featureScaling as featureScaling

from nn.models.NeuralNetwork import NeuralNetwork
from nn.components.connections.FullConnection import FullConnection
from nn.components.connections.Bias import Bias
from nn.components.layers.InputLayer import InputLayer
from nn.components.layers.SigmoidLayer import SigmoidLayer
from nn.components.layers.SoftmaxLayer import SoftmaxLayer
from nn.components.objectives.CrossEntropyObjective import CrossEntropyObjective
from trainers.SGD import SGDTrainer


training_percentage = 0.8

if __name__ == '__main__':
   # Load the data

   iris_data, iris_classes = load_iris_data()

   # Normalize the input data
   iris_data = featureScaling.mean_stdev(iris_data)

   # Split into training and test data
   idx = range(iris_data.shape[0])
   random.shuffle(idx)
   numVariables = 4
   numHidden = 5
   numClasses = 3

   training_set_X = iris_data[idx[:120],:]
   training_set_Y = iris_classes[idx[:120],:]
   test_set_X = iris_data[idx[120:],:]
   test_set_Y = iris_classes[idx[120:],:] 

   training_set = (training_set_X, training_set_Y)
   test_set = (test_set_X, test_set_Y)
 
#   training_set_X = gpu.garray(training_set_X)
#   training_set_Y = gpu.garray(training_set_Y)
#   test_set_X = gpu.garray(test_set_X)
#   test_set_Y = gpu.garray(test_set_Y)

   # Create the model
   print "Creating model...",
   input_layer = InputLayer(numVariables)
   hidden_layer = SigmoidLayer(numHidden)
   output_layer = SoftmaxLayer(numClasses)
   target_layer = InputLayer(numClasses)
   input_to_hidden = FullConnection(input_layer.output, hidden_layer.input)
   hidden_to_output = FullConnection(hidden_layer.output, output_layer.input)
   hidden_bias = Bias(hidden_layer.input)
   output_bias = Bias(output_layer.input)
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
   logger = ConsoleLogger(net, (training_set_X, training_set_Y), (test_set_X, test_set_Y))
   trainer = SGDTrainer(net, learning_rate=0.5, momentum=0.1, weight_decay=0.001, logger=logger)
   print "Done"

   print "Training..."
   for i in range(10000):
     trainer.trainBatch((training_set_X, training_set_Y))
     print "Iteration", i, "\tObjetive =", net.getObjective()
