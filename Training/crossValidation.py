## crossValidation.py    Dana Hughes    version 1.0     21-February-2015
##
## Functions to perform cross-validation of a model
##
## Revisions:
##   1.0   Initial version

import numpy as np


def k_fold_cross_validation(model, teacher, folds_X, folds_Y, logger = None):
   """
   Perform k-fold cross validation on the model
   """

   num_folds = len(folds_X)

   validation_costs = []
   validation_accuracies = []

   for i in range(num_folds):
      # Reset the model
      model.randomize_weights()

      validation_set_data = folds_X[i]
      validation_set_labels = folds_Y[i]

      training_set_X = []
      training_set_Y = []

      for j in range(num_folds):
         if i != j:
            training_set_X = training_set_X + folds_X[j]
            training_set_Y = training_set_Y + folds_Y[j]

      logger.set_training_data(training_set_X, training_set_Y)
      logger.set_test_data(validation_set_data, validation_set_labels)

      # Train the model

      teacher.train_minibatch(training_set_X, training_set_Y, 10)

      validation_costs.append(model.cost(validation_set_data, validation_set_labels))

      correct_predictions = 0
      for data, label in zip(validation_set_data, validation_set_labels):
         prediction = model.classify(data)
         if prediction == label:
            correct_predictions += 1

      validation_accuracies.append((100.0*correct_predictions) / len(validation_set_data))

      print "Fold",i,"-"
      print "  Cost:    ", validation_costs[i]
      print "  Correct: ", correct_predictions, 'of', len(validation_set_data)
      print "  Accuracy:", validation_accuracies[i]

   print "Cross-Validation results:"
   print "  Expected Cost:    ", np.mean(validation_costs)
   print "  Expected Accuracy:", np.mean(validation_accuracies)

   return np.mean(validation_costs), np.mean(validation_accuracies)
