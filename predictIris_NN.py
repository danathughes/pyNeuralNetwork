## predictIris.py
##
## Simple script to predict iris using neural network code
##

import random
import time
from datasets.iris import *
import Preprocess.featureScaling as featureScaling

from nn.models.NeuralNetwork import NeuralNetwork
from nn.components.connections.FullConnection import FullConnection
from nn.components.connections.Bias import Bias
from nn.components.layers.InputLayer import InputLayer
from nn.components.layers.SigmoidLayer import SigmoidLayer
from nn.components.layers.SoftmaxLayer import SoftmaxLayer
from nn.components.objectives.CrossEntropyObjective import CrossEntropyObjective
from trainers.SGD_flat import SGDTrainer
from trainers.PSO_flat import PSOTrainer

import matplotlib.pyplot as plt

training_percentage = 0.8

def run():
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

   SGD_objectives = []
   PSO_objectives = []

   print "Creating an SGD trainer..."
   trainer = SGDTrainer(net, learning_rate=0.9, momentum=0., weight_decay=0.001)

   print "Training..."
   start_time = time.time()
   for i in range(10000):
     trainer.trainBatch(training_set)
     net.evaluate(training_set)
#     print "Iteration", i, "\tObjetive =", trainer.global_best
#     print "Iteration", i, "\tObjetive =", net.getObjective()
     SGD_objectives.append(net.getObjective())

   SGD_time = time.time() - start_time

   net.randomize()

   trainer = PSOTrainer(net, number_particles=100, initial_weight_range=(-3.0,3.0), max_velocity = 0.1)
   print "Done"

   print "Training..."

   start_time = time.time()
   for i in range(100):
     trainer.trainBatch(training_set)
     net.evaluate(training_set)
#     print "Iteration", i, "\tObjetive =", trainer.global_best
#     print "Iteration", i, "\tObjetive =", net.getObjective()

     for i in range(100):
        PSO_objectives.append(net.getObjective())

   PSO_time = time.time() - start_time

#   net.evaluate(test_set)
#   print net.getOutput()
#   print net.getObjective()

   plt.plot(range(10000), SGD_objectives, '-b', range(10000), PSO_objectives, '-r')
   plt.legend(['SGD', 'PSO'])

   plt.xlabel('Number of Evaluations')
   plt.ylabel('Cross-Entropy Error')
   plt.show()

   print "SGD_time: ", SGD_time
   print "PSO_time: ", PSO_time

if __name__ == '__main__':
   run()
