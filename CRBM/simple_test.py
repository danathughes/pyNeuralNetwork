import numpy as np
import matplotlib.pyplot as plt

dataset = [[np.random.normal(1,0.2), np.random.normal(1,0.1)] for i in range(50)]
dataset = dataset + [[np.random.normal(-1,0.2), np.random.normal(-1,0.1)] for i in range(50)]

print dataset

XY = np.array(dataset)
plt.scatter(XY[:,0], XY[:,1])
plt.show()



