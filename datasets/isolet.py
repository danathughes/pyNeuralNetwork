def load_isolet(filename):
   """
   """

   f = open(filename, 'r')

   # We'll return a list of feature vectors and class vectors
   features = []
   classes = []

 
   for line in f.readlines():          # Just the instance name - unused
      data = [float(item) for item in line.strip().split(',')]
      features.append(data[:-1])
      label = [0]*26
      label[int(data[-1])-1] = 1
      classes.append(label)
 
   # All done!
   f.close()

   return features, classes

