## training.py    Dana Hughes    version 1.0     09-February-2015
##
## Functions used to train arbitrary models.  
##
## Revisions:
##   1.0   Initial version, modified algorithms from LogisticRegression

import numpy as np
import random

def train_epoch(model, data, output, learning_rate = 0.1):
   """
   Train once on each of the items in the provided dataset
   """

   gradient = np.array(model.gradient(data, output))
   model.update_weights(-learning_rate * gradient)



