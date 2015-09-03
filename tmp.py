from objectives.MSEObjective import *
from layers.InputLayer import *
from layers.SigmoidLayer import *
from layers.TanhLayer import *
from layers.SoftmaxLayer import *
from connections.FullConnection import *
from datasets.iris import *

dataset, targets = load_iris_data()

input_layer = InputLayer(4)
target_layer = InputLayer(3)
conn1 = FullConnection(4,5)
hidden_layer = TanhLayer()
conn2 = FullConnection(5,3)
output_layer = SoftmaxLayer()
objective = MSEObjective()

conn1.setInputConnection(input_layer)
conn1.setOutputConnection(hidden_layer)
conn2.setInputConnection(hidden_layer)
conn2.setOutputConnection(output_layer)
input_layer.output_connections.append(conn1)
hidden_layer.input_connections.append(conn1)
hidden_layer.output_connections.append(conn2)
output_layer.input_connections.append(conn2)
output_layer.output_connections.append(objective)

objective.setOutputLayer(output_layer)
objective.setTargetLayer(target_layer)

input_layer.setInput(dataset)
target_layer.setInput(targets)

conn1.randomize()
conn2.randomize()

def fwd():
   input_layer.forward()
   target_layer.forward()
   conn1.forward()
   hidden_layer.forward()
   conn2.forward()
   output_layer.forward()
   objective.forward()

def bwd():
   objective.backward()
   output_layer.backward()
   conn2.backward()
   hidden_layer.backward()
   conn1.backward()
   target_layer.backward()
   input_layer.backward()

for i in range(10000):
   conn1.reset()
   conn2.reset()
   fwd()
   print i, objective.getObjective()
   bwd()
   conn1.updateParameterGradient()
   g = conn1.getParameterGradient()
   conn1.updateParameters(0.9*g/150)
   conn2.updateParameterGradient()
   g = conn2.getParameterGradient()
   conn2.updateParameters(0.9*g/150)

   

