TODO
====

* 20-Feb-2015

* * Maybe make UML class diagram or something

* 17-Feb-2015

* * Work on the Training module - perhaps have a single Teacher class with a couple of lists of functions

* * * Update the weights - gradient, momentum, weight reduction / normalization, etc.

* * * Stopping criteria - minimized error, change in error, etc.

* * Modify LinearRegression model to match NeuralNetwork and LogisticRegression

* * Merge LinearRegression and LogisticRegression into one model?

* * * LinearRegression == LogisticRegression with square error function and linear activation function

* * * RegressionModel, SimpleLinearRegression, SimpleLogisticRegression, etc.?

* * Logger ideas:

* * * Real time graphics update - plot cost vs. iteration

* * * Test / validation subset classification - plot predictions for a set of test data

* * * Log to a file

* * * Composite logger - has multiple loggers and delegates to each

* * * Plot weights / bias functions (visualize receptive field)

* 16-Feb-2015

* * Make Logger class which we can subclass from--perhaps have the model, training and test data as class attributes, so that nothing needs to be passed when the log methods are called.  Or would a Listener pattern be better?  

* * Have a function (functions?) to create confusion matrices

* * Look into statistical tests like ROC, AUC, F-score, etc.  These may be good to put into a logger class, or have as external functions called by individual logger classes.

* * matplotlib outupt?


