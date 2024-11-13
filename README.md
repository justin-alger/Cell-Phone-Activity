# Cell-Phone-Activity
Data was obtained recording phone sensor readings from cell phone users in 6 different activity modes. There is a total of
n=165,633 readings with 1 binary, 2 nominal and 16 interval attributes.

The target is 'activity', a nominal attribute with 5 classes. The objective is to identify the activity of the phone 
user from phone sensor readings and user descriptions.
The goal was to develop and evaluate two models:
1. Logistic Regression with L2 Regularization
2. Decision Tree
I optimize the parameters for these models using 10-fold cross validation.

The model parameters used for hyperparameter optimization are:
1. L2 Logistic Regression: C
2. Decision Tree: max_depth, min_samples_leaf and min_samples_split.
This is a large dataset, and in practice we could use more than 10 folds. However, runtime increases
as the number of folds increases so I used 10.

Since the target is nominal, we need to use an appropriate metric.
For a binary target we used ‘f1’. Sklearn does not accept that metric
when the target is nominal. Instead is accepts either ‘f1_micro’ or
‘f1_macro’. The user documentation in sklearn describes the difference,
but for this I use ‘f1_micro’.
