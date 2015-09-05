from nn.components.objectives.CrossEntropyObjective import *
from nn.components.objectives.MSEObjective import *
from nn.components.layers.InputLayer import *
from nn.components.layers.SigmoidLayer import *
from nn.components.layers.TanhLayer import *
from nn.components.layers.ReluLayer import *
from nn.components.layers.SoftReluLayer import *
from nn.components.layers.SoftmaxLayer import *
from nn.components.connections.FullConnection import *
from nn.components.connections.Bias import *
from nn.models.NeuralNetwork import NeuralNetwork
from datasets.iris import *

dataset, targets = load_iris_data()

# Create layers
input_layer = InputLayer(4)
target_layer = InputLayer(3)
hidden_layer = SoftReluLayer(5)
output_layer = SoftmaxLayer(3)

# Make the connection from input to output
conn1 = FullConnection(input_layer.output, hidden_layer.input)
bias1 = Bias(hidden_layer.input)
conn2 = FullConnection(hidden_layer.output, output_layer.input)
bias2 = Bias(output_layer.input)

# Make the objective
objective = MSEObjective(output_layer.output, target_layer.output)

# Build the neural network
net = NeuralNetwork()
net.addLayer(input_layer)
net.addLayer(target_layer)
net.addConnection(conn1)
net.addConnection(bias1)
net.addLayer(hidden_layer)
net.addConnection(conn2)
net.addConnection(bias2)
net.addLayer(output_layer)
net.addObjective(objective)

# Set the input and targets
net.setInputLayer(input_layer)
net.setTargetLayer(target_layer)
net.setObjective(objective)
net.setOutputLayer(output_layer)

# Set the input and targets to train to
net.setInput(dataset)
net.setTarget(targets)

net.randomize()

for i in range(10):

   # Do a training cycle
   net.reset()
   net.forward()
   net.backward()
   net.update()

   print i, net.getObjective()

   # Get the gradients
   gradients = net.getParameterGradients()
   updates = {}
   # Multiply by the learning rate
   for connection, gradient in gradients.items():
      updates[connection] = -0.9*gradient / 150

   # Update the gradients
   net.updateParameters(updates)
   

