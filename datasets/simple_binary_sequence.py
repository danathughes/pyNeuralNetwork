

def get_sequence():
   """
   Create a simple binary sequence which switches between two patterns
   """

   values = [[1,0,1],[0,1,1],[1,0,0]]

   pattern = [0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,0,2,2,0,0,2,2,0,0,2,2,0,0,2,2,1,1,2,0,1,1,2,0,1,1,2,0,0,2,2,0,0,1,2,0,1,2,0,0,2,2,0,0,2,2,1,1,2,0,1,1]

   sequence = [values[p] for p in pattern]

   return sequence, pattern 
