#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Oct 2024
@author: Justin Alger
"""

import pandas as pd
import numpy as  np

from AdvancedAnalytics.ReplaceImputeEncode import DT, ReplaceImputeEncode
from AdvancedAnalytics.Tree                import tree_classifier
from AdvancedAnalytics.Regression          import logreg
from sklearn.tree            import DecisionTreeClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.linear_model    import LogisticRegressionCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, make_scorer, mean_squared_error
import statsmodels.api as sm
import matplotlib.pyplot as plt
from copy import deepcopy

df = pd.read_excel("CellphoneActivity.xlsx")
print("These data contain", df.shape[0], "obs with", df.shape[1], 
      "attributes.")
print("Below are the first 5 obs:")
print(df.head(5))
print("Now, resampling to ensure data is randomized:\n")
df = df.sample(frac=1, random_state=12345)
print("The randomized data contains", df.shape[0], "obs with", df.shape[1], 
      "attributes.")
print("Below are the first 5 obs:")
print(df.head(5))

attribute_map = {
    'activity':[DT.Ignore,('sitting', 'sittingdown', 'standing',
                            'standingup', 'walking')],
    'user':    [DT.Nominal,('debora', 'jose_carlos', 'katia', 'wallace')],
    'gender':  [DT.Binary, ('Man', 'Woman')],
    'age':     [DT.Interval,(20, 80)],
    'height':  [DT.Interval,(1.5, 1.75)],
    'weight':  [DT.Interval,(50, 85)],
    'BMI':     [DT.Interval,(20, 30)],
    'x1':      [DT.Interval,(-750, +750)],
    'y1':      [DT.Interval,(-750, +750)],
    'z1':      [DT.Interval,(-750, +750)],
    'x2':      [DT.Interval,(-750, +750)],
    'y2':      [DT.Interval,(-750, +750)],
    'z2':      [DT.Interval,(-750, +750)],
    'x3':      [DT.Interval,(-750, +750)],
    'y3':      [DT.Interval,(-750, +750)],
    'z3':      [DT.Interval,(-750, +750)],
    'x4':      [DT.Interval,(-750, +750)],
    'y4':      [DT.Interval,(-750, +750)],
    'z4':      [DT.Interval,(-750, +750)]
}

""" ************************************************************************ """
"""    DATA PREPROCESSING                                                    """
""" ************************************************************************ """
print("\n********************************************************************")
print("***                  DATA PREPROCESSING                          ***")
target = "activity"
rie = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding='one-hot', 
                          binary_encoding='one-hot', drop=False, display=True) 
#drop=False since we are using regularization in the Logistic regression
encoded_df = rie.fit_transform(df)
#print(encoded_df.head())  #to observe the X df after encoding
#Binary one-hot encoding of gender results in Man=0 and Woman=1
y = df[target] 
#print(y.head()) #just to test and see the y df

X = encoded_df

Xt, Xv, yt, yv = train_test_split(X, y, train_size=0.9, random_state=12345, stratify=y)
""" ************************************************************************ """
"""  L2 LOGISTIC REGRESSION HYPERPARAMETER OPTIMIZATION                       """
""" ************************************************************************ """
def lr_plot(lr, X, y):
    #C_vals = np.logspace(-3, 2, 10)
    #C_vals=[1e-4, 1e-3, 1e-2, 1, 1e2, 1e4]
    C_vals = C_list
    coefs  = []
    f1_micro_scores    = []
    for c in C_vals:
        lr.set_params(Cs=[c])
        lr.fit(X, y)
        coefs.append(lr.coef_.ravel()) #By using .ravel(), I flatten the coefficients and ensure that coefs is a list of 1D arrays, which is easier to handle for plotting and comparison across different C values.
        predictions = lr.predict(X)
        f1 = f1_score(y,predictions, average='micro')  
        f1_micro_scores.append(f1)
        print("C: ", c, "F1-micro: ", f1)
        print("Starting new C loop.")
    # Find the best C value (corresponding to the highest avg F1-micro score):
    best_index = np.argmax(f1_micro_scores)
    best_C = C_vals[best_index]
    best_f1_micro = f1_micro_scores[best_index]
    # Print the best C value and the corresponding F1-micro score
    print(f"Best C value: {best_C}")
    print(f"Best F1-micro score: {best_f1_micro}")    
    
    plt.figure(figsize=(10, 3))
    plt.subplot(121)
    ax = plt.gca()
    ax.plot(C_vals, coefs)
    ax.set_xscale('log')
    plt.xlabel('C')
    plt.ylabel('L2 Coefficients')
    plt.title('L2 Coefficients vs. C (Reg. Strength)')
    plt.axis('tight')
    
    plt.subplot(122)
    ax2 = plt.gca()
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')
    ax2.plot(C_vals, f1_micro_scores)
    ax2.set_xscale('log')
    plt.xlabel('C')
    plt.ylabel('F1-icro Score')
    plt.title('f1_micro vs. C (Reg. Strength)')
    plt.axis('tight')
    plt.show()

    return best_C, best_f1_micro

print("\n********************************************************************")
print("**********  SKLEARN LOGISTIC REG CV - Target: 'activity'    *********")
print("**********  L2 Regularization LOGISTIC REGRESSION CV       *********")
print("**********  C is 1/alpha, the inverse of shrinkage         *********")
print("********************************************************************")

# Used https://scikit-learn.org/dev/modules/generated/sklearn.linear_model.LogisticRegressionCV.html to
# build the logistic regression with cross-validation and hyperparameter optimization:
le = LabelEncoder() # Convert target column to numeric labels using LabelEncoder
df['target_numeric'] = le.fit_transform(df[target])

C_list=[0.0001, 0.001, 0.01, 0.05, 0.1, 1, 10, 100]

lr_cv = LogisticRegressionCV(
    #Cs=10,                 # Use int to try # values b/w 1e-4 and 1e4, or can enter a list of specific values.
    Cs=C_list,                 # List of specific values. 
    cv=10,                    # 10-fold cross-validation
    penalty='l2',              # L2 regularization
    scoring='f1_micro',        # Use f1_micro as scoring metric
    solver='lbfgs',            # lbfgs solver for multinomial; For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss
    multi_class='multinomial', # Multinomial logistic regression because target is nominal, not binary
    max_iter=10000,             # Tried 100, 200, 1000, 2000 and did not converge, 10000 seemed to work
    random_state=12345,         # Set random state for reproducibility
    n_jobs=5
)
# Fit the model
lr_cv.fit(Xt, yt)

print("Options tested by lr_cv for C values: ", lr_cv.Cs_)
# Get the best C value chosen by the CV model
print(f"Best C parameter in the lr_cv model: {lr_cv.C_[0]}")
'''Per sklearn, C_ is an array of C that maps to the best scores across every class. If refit is set to False, 
then for each class, the best C is the average of the C’s that correspond to the best scores 
for each fold. C_ is of shape(n_classes,) when the problem is binary.'''

print("Models converged in ", lr_cv.n_iter_," iterations (list) with ", lr_cv.n_features_in_,"features.\n")
print(lr_cv.get_params)
yv_pred = lr_cv.predict(Xv)
# Generate a classification report for lr_cv
print("Classification Report (Validation Set):\n", classification_report(yv, yv_pred, target_names=le.classes_))
logreg.display_split_metrics(lr_cv, Xt, yt, Xv, yv)

print("Printing list of attributes w/ coefficients:\n")
lrcv_l2_selected = []
i = 0
lrcv_l2_coef = lr_cv.coef_[0]
for coef in lrcv_l2_coef:
    if abs(coef)>0.0005:
        lrcv_l2_selected.append(X.columns[i])
    i += 1
print("L2 w/ CV SELECTED ", len(lrcv_l2_selected), "ATTRIBUTES")
logreg.display_coef(lr_cv, Xt, yt)

""" ************************************************************************ """
"""               DECISION TREE HYPERPARAMETER OPTIMIZATION                  """
""" ************************************************************************ """
print("\n********************************************************************")
print("*******  SKLEARN Decision Tree Classifier - Target: 'activity' *******")
print("********************************************************************")

candidate_depths = [6, 9, 12, 15, 18] 
candidate_splits = [2, 3, 4]
candidate_leafs  = [1, 2, 3]
best_metric      = 0.0 #the metric is F1 which is between 0 and 1
metric           = 'f1_micro'
n_folds=10

for depth in candidate_depths:
    for split in candidate_splits:
        for leaf in candidate_leafs:
            dt = DecisionTreeClassifier(max_depth=depth, 
                                       min_samples_split=split, 
                                       min_samples_leaf=leaf,
                                       random_state=12345)
            score = cross_val_score(dt, X, y, cv=n_folds, scoring=metric)
            f1    = np.mean(score)
            std   = np.std(score)
            if f1>best_metric:
                best_metric = f1
                best_tree   = deepcopy(dt)
                print("\nDecision Tree for Depth=", depth, "Min Split=", split,
                      "Min Leaf Size=", leaf)
                print("{:.<18s}{:>6s}{:>13s}".format("Metric", 
                                                     "Mean", "Std. Dev."))
                print("{:.<18s}{:>7.4f}{:>10.4f}".format("F1 Micro: ", f1, std))

best_tree.fit(Xt, yt)
tree_levels = best_tree.get_depth()
tree_leafs  = best_tree.tree_.n_leaves 
print("\nThe Best Tree has", tree_levels, "Levels and", 
      tree_leafs, "Leaves")
print("\nBEST DECISION TREE SELECTED USING ", n_folds, 
      "-FOLD CV WITH F1 Micro=", round(best_metric, 3))
tree_parms = best_tree.get_params()
print("Max Depth=", tree_parms['max_depth'])
print("Min Samples Split=", tree_parms['min_samples_split'])
print("Min Samples Leaf=", tree_parms['min_samples_leaf'])
tree_classifier.display_importance(best_tree, Xt.columns, top=20, plot=True)
tree_classifier.display_split_metrics(best_tree, Xt, yt, Xv, yv)