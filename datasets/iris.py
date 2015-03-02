## iris.py
##
## Helper file to load iris data

irisDataFileName = '/home/dana/Desktop/Research/deepLearning/datasets/data/iris.data'

class_names = {'Iris-setosa':     [1,0,0], 
               'Iris-versicolor': [0,1,0],
               'Iris-virginica':  [0,0,1]
              } 

def load_iris_data(filename = irisDataFileName):
   """
   Load the iris dataset
   """

   iris_data_file = open(irisDataFileName)
   iris_data = []
   iris_classes = []

   for data in iris_data_file.readlines():
      iris_data.append([eval(num) for num in data.split(',')[0:4]])
      flower = data.split(',')[-1].strip()
      iris_classes.append(class_names[flower])

   iris_data_file.close()

   return iris_data, iris_classes
