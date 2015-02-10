from Tkinter import *

def load_digits(filename):
   """
   """

   f = open(filename, 'r')

   # We'll return a list of digit vectors and class vectors
   digits = []
   classes = []

   # Loop through 
 
   while f.readline():          # Just the instance name - unused
      # The next 14 lines indicate a new digit
      # Flatten this to a 196 element vector
      digit = []                   
      for i in range(14):
         line = f.readline()
         digit = digit + [float(data) for data in line.split()]
      digits.append(digit)

      # The last line is the classes
      line = f.readline()
      digit_class = [int(data) for data in line.split()]
      classes.append(digit_class)

   # All done!
   f.close()

   return digits, classes

   
def show_digit(digit):
   """
   Create a window and show the digit
   """

   # Create a window for the digit.  The digit is 14x14, so create a window 
   # which is 150x150.  We'll leave a border of 5 pixels, and each digit
   # "pixel" will be 10x10

   master = Tk()

   canvas = Canvas(master, width=150, height=150)
   canvas.pack()

   # Draw a rectange for each pixel in the digit
   for i in range(14):
      y = 10*i + 5
      for j in range(14):
         x = 10*j + 5
         

         # Determine the hex value of this pixel color
         pixel_value = digit[14*i + j]
         pixel_hex = hex(int(pixel_value*255)).replace('0x','')
         pixel_hex = '#' + pixel_hex + pixel_hex + pixel_hex
         
         # Draw the rectangle
         canvas.create_rectangle(x, y, x+10, y+10, fill=pixel_hex)

   # Done!
   return canvas


def digit_gray_to_binary(digit, threshold = 0.5):
   """
   """

   # Set the digit to 1 if greater than the threshold, 0 otherwise
   return [1.0 if pixel >= threshold else 0.0 for pixel in digit]
