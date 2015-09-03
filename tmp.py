from nn.components.objectives.CrossEntropyObjective import *
from nn.components.layers.InputLayer import *
from nn.components.layers.SigmoidLayer import *
from nn.components.layers.TanhLayer import *
from nn.components.layers.SoftmaxLayer import *
from nn.components.connections.FullConnection import *
from nn.components.connections.Bias import *
from datasets.iris import *

dataset, targets = load_iris_data()

input_layer = InputLayer(4)
target_layer = InputLayer(3)
conn1 = FullConnection(4,5)
bias1 = Bias(5)
hidden_layer = TanhLayer()
conn2 = FullConnection(5,3)
bias2 = Bias(3)
output_layer = SoftmaxLayer()
objective = CrossEntropyObjective()

conn1.setFromLayer(input_layer)
conn1.setToLayer(hidden_layer)
conn2.setFromLayer(hidden_layer)
conn2.setToLayer(output_layer)
bias1.setToLayer(hidden_layer)
bias2.setToLayer(output_layer)

input_layer.output_connections.append(conn1)
hidden_layer.input_connections.append(conn1)
hidden_layer.input_connections.append(bias1)
hidden_layer.output_connections.append(conn2)
output_layer.input_connections.append(conn2)
output_layer.input_connections.append(bias2)
output_layer.output_connections.append(objective)

objective.setOutputLayer(output_layer)
objective.setTargetLayer(target_layer)

input_layer.setInput(dataset)
target_layer.setInput(targets)

conn1.randomize()
conn2.randomize()
bias1.randomize()
bias2.randomize()

def fwd():
   input_layer.forward()
   target_layer.forward()
   conn1.forward()
   bias1.forward()
   hidden_layer.forward()
   conn2.forward()
   bias2.forward()
   output_layer.forward()
   objective.forward()

def bwd():
   objective.backward()
   output_layer.backward()
   conn2.backward()
   bias2.backward()
   hidden_layer.backward()
   conn1.backward()
   bias1.backward()
   target_layer.backward()
   input_layer.backward()

for i in range(10000):
   conn1.reset()
   conn2.reset()
   bias1.reset()
   bias2.reset()
   fwd()
   print i, objective.getObjective()
   bwd()
   conn1.updateParameterGradient()
   g = conn1.getParameterGradient()
   conn1.updateParameters(0.5*g/150)
   conn2.updateParameterGradient()
   g = conn2.getParameterGradient()
   conn2.updateParameters(0.5*g/150)
   bias1.updateParameterGradient()
   g = bias1.getParameterGradient()
   bias1.updateParameters(0.5*g/150)
   bias2.updateParameterGradient()
   g = bias2.getParameterGradient()
   bias2.updateParameters(0.5*g/150)
   
   
print output_layer.getOutput()
