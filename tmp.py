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

# Create a neural network
net = NeuralNetwork()

# Create layers
input_layer = InputLayer(4)
target_layer = InputLayer(3)
hidden_layer = TanhLayer(5)
output_layer = SoftmaxLayer(3)

# Make the connection from input to output
conn1 = FullConnection(input_layer.output, hidden_layer.input)
bias1 = Bias(hidden_layer.input)
conn2 = FullConnection(hidden_layer.output, output_layer.input)
bias2 = Bias(output_layer.input)

# Add the objective
objective = CrossEntropyObjective(output_layer.output, target_layer.output)

net.addLayer(input_layer)
net.addLayer(target_layer)
net.addConnection(conn1)
net.addConnection(bias1)
net.addLayer(hidden_layer)
net.addConnection(conn2)
net.addConnection(bias2)
net.addLayer(output_layer)
net.addObjective(objective)

net.setInputLayer(input_layer)
net.setTargetLayer(target_layer)
net.setInput(dataset)
net.setTarget(targets)
net.setOutputLayer(output_layer)
net.setObjective(objective)

# Randomize these connections
net.randomize()

# Set the input and targets
# Everythin from here on out deals directly with net

for i in range(10000):
   net.reset()
   net.forward()
   print i, net.getObjective()
   net.backward()

   net.update()

   gradients = net.getParameterGradients()
   update = {}

   for conn, grad in gradients.items():
      update[conn] = 0.1*grad/150
      
   net.updateParameters(update)

print net.getOutput()
