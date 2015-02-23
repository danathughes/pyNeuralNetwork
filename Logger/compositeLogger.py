## compositeLogger.py
##
## Log data to multiple loggers
##

import numpy as np

class CompositeLogger:
   """
   """

   def __init__(self, model, training_data=([], []), test_data=([], []), validation_data=([], [])):
      """
      """

      self.model = model
 
      self.training_data = training_data[0]
      self.training_labels = training_data[1]

      self.test_data = test_data[0]
      self.test_labels = test_data[1]

      self.validation_data = validation_data[0]
      self.validation_labels = validation_data[1]

      self.loggers = []


   def add_logger(self, logger):
      """
      """

      self.loggers.append(logger)



   def log_setup(self):
      """

      """

      for logger in self.loggers:
         logger.log_setup()


   def log_training(self, epoch_number):
      """

      """

      for logger in self.loggers:
         logger.log_training(epoch_number)


   def log_results(self):
      """

      """

      for logger in self.loggers:
         logger.log_results()

