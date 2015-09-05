from NeuralNetwork.neuralNetwork import *
from Functions.functions import *
from Training.teacher import *
from Logger.consoleLogger import *
from Logger.graphLogger import *
from Logger.compositeLogger import *
import sys


def load_data(filename):
   """
   """

   class_labels = {'"no_touch"':[1,0,0,0], 
                   '"tap"':	[0,1,0,0],
                   '"up"':	[0,0,1,0],
                   '"down"':	[0,0,0,1]}


   f = open(filename)

   dataset = []
   labels = []

   f.readline()

   for line in f.readlines():
      data = line.strip().split(',')

      values = [float(d) for d in data[:10]]
      label = class_labels[data[-1]]

      if data[-1] != '"no_touch"':
         dataset.append(values)
         labels.append(label)

   return dataset, labels


def main():
   """
   """

   dataset, labels = load_data("training_data.csv")

   num_input = len(dataset[0])
   num_output = len(labels[0])
   num_hidden = 10

   NN = NeuralNetwork([num_input, num_hidden, num_output], [None, SIGMOID, SOFTMAX], CROSS_ENTROPY)

   graphLogger = GraphLogger(NN, (dataset, labels), (dataset, labels))
   consoleLogger = ConsoleLogger(NN, (dataset, labels), (dataset, labels))
   logger = CompositeLogger()
   logger.add_logger(graphLogger)
   logger.add_logger(consoleLogger)

   teacher = Teacher(NN, logger)

   teacher.add_weight_update(0.5, gradient_descent)
   teacher.add_weight_update(0.1, momentum)
   teacher.add_weight_update(0.00001, weight_decay)

   teacher.train_minibatch(dataset, labels, 20)

   logger.log_results()


if __name__=="__main__":
   main()

