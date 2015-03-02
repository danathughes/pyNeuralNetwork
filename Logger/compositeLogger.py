## compositeLogger.py
##
## Log data to multiple loggers
##

import numpy as np

class CompositeLogger:
   """
   """

   def __init__(self):
      """
      """

      self.loggers = []

   def set_training_data(self, data, labels):
      for logger in self.loggers:
         logger.set_training_data(data, labels)

   def set_test_data(self, data, labels):
      for logger in self.loggers:
         logger.set_test_data(data, labels)


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

