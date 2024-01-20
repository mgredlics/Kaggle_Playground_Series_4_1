#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 14:26:16 2024


Binary Classification with Bank Churn Dataset
Kaggle Playground Series - Season 4, Episode 1

@author: mgredlics
"""


import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
from yellowbrick.classifier import ConfusionMatrix
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from scipy.stats import randint,uniform
from sklearn.inspection import permutation_importance
import xgboost as xgb



### Model Building
# Read in Preprocessed Files

X_train_scaled = pd.read_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/PreProcessedData/X_train_scaled.csv",index_col=0)
X_test_scaled = pd.read_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/PreProcessedData/X_test_scaled.csv", index_col=0)
y_train = pd.read_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/PreProcessedData/y_train.csv", index_col=0)
y_test = pd.read_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/PreProcessedData/y_test.csv", index_col=0)

submissionX_scaled = pd.read_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/PreProcessedData/submissionX_scaled.csv", index_col=0)
submission_id = pd.read_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/PreProcessedData/submission_id.csv", index_col=0)



## Fit Models
# xgboost - tuned_1 - Fix learning rate and number of estimators
xgb_model = xgb.XGBClassifier(objective="binary:logistic", nthread=4, scale_pos_weight=1,random_state=42)

param_dist = {'max_depth': [5],
              'learning_rate': [0.1],
              'subsample': [0.8],
              'n_estimators':[1000],
              'min_child_weight': [1],
              "gamma": [0],
              "colsample_bytree": [0.8]} 


grid_search = GridSearchCV(xgb_model, 
                                 param_grid= param_dist, 
                                 cv=5,
                                 scoring="roc_auc")


grid_search.fit(X_train_scaled, y_train)

# Create a variable for the best model
best_xgb_tuned = grid_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  grid_search.best_params_)

# Generate predictions with the best model
y_pred_xgb_tuned = best_xgb_tuned.predict(X_test_scaled)

# Confusion Matrix

conf_matrix_xgb_tuned = metrics.confusion_matrix(y_test, y_pred_xgb_tuned)
conf_matrix_xgb_tuned

print(metrics.classification_report(y_test, y_pred_xgb_tuned))


y_pred_proba_xgb_tuned = best_xgb_tuned.predict_proba(X_test_scaled)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_xgb_tuned)
auc = metrics.roc_auc_score(y_test, y_pred_proba_xgb_tuned)
plt.plot(fpr,tpr,label="XGB - Tuned - Step 1, auc on test set="+str(auc))
plt.legend(loc=4)
plt.show()


## Fit Models
# xgboost - tuned_2 - Tune max depth and min child weight
xgb_model = xgb.XGBClassifier(objective="binary:logistic", nthread=4, scale_pos_weight=1,random_state=42)

param_dist = {'max_depth': range(3,10,2),
              'learning_rate': [0.1],
              'subsample': [0.8],
              'n_estimators':[1000],
              'min_child_weight': range(1,6,2),
              "gamma": [0],
              "colsample_bytree": [0.8]} 


grid_search = GridSearchCV(xgb_model, 
                                 param_grid= param_dist, 
                                 cv=5,
                                 scoring="roc_auc")


grid_search.fit(X_train_scaled, y_train)

# Look at details


# Create a variable for the best model
best_xgb_tuned = grid_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  grid_search.best_params_)

# Generate predictions with the best model
y_pred_xgb_tuned = best_xgb_tuned.predict(X_test_scaled)

# Confusion Matrix

conf_matrix_xgb_tuned = metrics.confusion_matrix(y_test, y_pred_xgb_tuned)
conf_matrix_xgb_tuned

print(metrics.classification_report(y_test, y_pred_xgb_tuned))


y_pred_proba_xgb_tuned = best_xgb_tuned.predict_proba(X_test_scaled)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_xgb_tuned)
auc = metrics.roc_auc_score(y_test, y_pred_proba_xgb_tuned)
plt.plot(fpr,tpr,label="XGB - Tuned - Step 2, auc on test set="+str(auc))
plt.legend(loc=4)
plt.show()

# xgboost - tuned_3 - Final Tune max depth and min child weight
xgb_model = xgb.XGBClassifier(objective="binary:logistic", nthread=4, scale_pos_weight=1,random_state=42)

param_dist = {'max_depth': [1,2,3,4],
              'learning_rate': [0.1],
              'subsample': [0.8],
              'n_estimators':[1000],
              'min_child_weight': [2,3,4],
              "gamma": [0],
              "colsample_bytree": [0.8]} 


grid_search = GridSearchCV(xgb_model, 
                                 param_grid= param_dist, 
                                 cv=5,
                                 scoring="roc_auc")


grid_search.fit(X_train_scaled, y_train)

# Look at details


# Create a variable for the best model
best_xgb_tuned = grid_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  grid_search.best_params_)

# Generate predictions with the best model
y_pred_xgb_tuned = best_xgb_tuned.predict(X_test_scaled)

# Confusion Matrix

conf_matrix_xgb_tuned = metrics.confusion_matrix(y_test, y_pred_xgb_tuned)
conf_matrix_xgb_tuned

print(metrics.classification_report(y_test, y_pred_xgb_tuned))


y_pred_proba_xgb_tuned = best_xgb_tuned.predict_proba(X_test_scaled)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_xgb_tuned)
auc = metrics.roc_auc_score(y_test, y_pred_proba_xgb_tuned)
plt.plot(fpr,tpr,label="XGB - Tuned - Step 3, auc on test set="+str(auc))
plt.legend(loc=4)
plt.show()


# xgboost - tuned_4 - Tune Gamma
xgb_model = xgb.XGBClassifier(objective="binary:logistic", nthread=4, scale_pos_weight=1,random_state=42)

param_dist = {'max_depth': [2],
              'learning_rate': [0.1],
              'subsample': [0.8],
              'n_estimators':[1000],
              'min_child_weight': [4],
              "gamma": [i/10.0 for i in range(0,5)],
              "colsample_bytree": [0.8]} 


grid_search = GridSearchCV(xgb_model, 
                                 param_grid= param_dist, 
                                 cv=5,
                                 scoring="roc_auc")


grid_search.fit(X_train_scaled, y_train)

# Look at details


# Create a variable for the best model
best_xgb_tuned = grid_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  grid_search.best_params_)

# Generate predictions with the best model
y_pred_xgb_tuned = best_xgb_tuned.predict(X_test_scaled)

# Confusion Matrix

conf_matrix_xgb_tuned = metrics.confusion_matrix(y_test, y_pred_xgb_tuned)
conf_matrix_xgb_tuned

print(metrics.classification_report(y_test, y_pred_xgb_tuned))


y_pred_proba_xgb_tuned = best_xgb_tuned.predict_proba(X_test_scaled)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_xgb_tuned)
auc = metrics.roc_auc_score(y_test, y_pred_proba_xgb_tuned)
plt.plot(fpr,tpr,label="XGB - Tuned - Step 4, auc on test set="+str(auc))
plt.legend(loc=4)
plt.show()


# xgboost - tuned_5 - Tune Sub Sample and colsample_bytree
xgb_model = xgb.XGBClassifier(objective="binary:logistic", nthread=4, scale_pos_weight=1,random_state=42)

param_dist = {'max_depth': [2],
              'learning_rate': [0.1],
              'subsample': [i/10.0 for i in range(6,10)],
              'n_estimators':[1000],
              'min_child_weight': [4],
              "gamma": [0.1],
              "colsample_bytree": [i/10.0 for i in range(6,10)]} 


grid_search = GridSearchCV(xgb_model, 
                                 param_grid= param_dist, 
                                 cv=5,
                                 scoring="roc_auc")


grid_search.fit(X_train_scaled, y_train)

# Look at details


# Create a variable for the best model
best_xgb_tuned = grid_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  grid_search.best_params_)

# Generate predictions with the best model
y_pred_xgb_tuned = best_xgb_tuned.predict(X_test_scaled)

# Confusion Matrix

conf_matrix_xgb_tuned = metrics.confusion_matrix(y_test, y_pred_xgb_tuned)
conf_matrix_xgb_tuned

print(metrics.classification_report(y_test, y_pred_xgb_tuned))


y_pred_proba_xgb_tuned = best_xgb_tuned.predict_proba(X_test_scaled)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_xgb_tuned)
auc = metrics.roc_auc_score(y_test, y_pred_proba_xgb_tuned)
plt.plot(fpr,tpr,label="XGB - Tuned - Step 5, auc on test set="+str(auc))
plt.legend(loc=4)
plt.show()


# xgboost - tuned_6 - Fine Tune Sub Sample and colsample_bytree
xgb_model = xgb.XGBClassifier(objective="binary:logistic", nthread=4, scale_pos_weight=1,random_state=42)

param_dist = {'max_depth': [2],
              'learning_rate': [0.1],
              'subsample': [i/100.0 for i in range(85,105,5)],
              'n_estimators':[1000],
              'min_child_weight': [4],
              "gamma": [0.1],
              "colsample_bytree": [i/100.0 for i in range(65,80,5)]} 


grid_search = GridSearchCV(xgb_model, 
                                 param_grid= param_dist, 
                                 cv=5,
                                 scoring="roc_auc")


grid_search.fit(X_train_scaled, y_train)

# Look at details


# Create a variable for the best model
best_xgb_tuned = grid_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  grid_search.best_params_)

# Generate predictions with the best model
y_pred_xgb_tuned = best_xgb_tuned.predict(X_test_scaled)

# Confusion Matrix

conf_matrix_xgb_tuned = metrics.confusion_matrix(y_test, y_pred_xgb_tuned)
conf_matrix_xgb_tuned

print(metrics.classification_report(y_test, y_pred_xgb_tuned))


y_pred_proba_xgb_tuned = best_xgb_tuned.predict_proba(X_test_scaled)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_xgb_tuned)
auc = metrics.roc_auc_score(y_test, y_pred_proba_xgb_tuned)
plt.plot(fpr,tpr,label="XGB - Tuned - Step 6, auc on test set="+str(auc))
plt.legend(loc=4)
plt.show()



# xgboost - tuned_7 - Tune Regularization Parapebers
xgb_model = xgb.XGBClassifier(objective="binary:logistic", nthread=4, scale_pos_weight=1,random_state=42)

param_dist = {'max_depth': [2],
              'learning_rate': [0.1],
              'subsample': [0.95],
              'n_estimators':[1000],
              'min_child_weight': [4],
              "gamma": [0.1],
              "colsample_bytree": [0.7],
              'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]} 


grid_search = GridSearchCV(xgb_model, 
                                 param_grid= param_dist, 
                                 cv=5,
                                 scoring="roc_auc")


grid_search.fit(X_train_scaled, y_train)

# Look at details


# Create a variable for the best model
best_xgb_tuned = grid_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  grid_search.best_params_)

# Generate predictions with the best model
y_pred_xgb_tuned = best_xgb_tuned.predict(X_test_scaled)

# Confusion Matrix

conf_matrix_xgb_tuned = metrics.confusion_matrix(y_test, y_pred_xgb_tuned)
conf_matrix_xgb_tuned

print(metrics.classification_report(y_test, y_pred_xgb_tuned))


y_pred_proba_xgb_tuned = best_xgb_tuned.predict_proba(X_test_scaled)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_xgb_tuned)
auc = metrics.roc_auc_score(y_test, y_pred_proba_xgb_tuned)
plt.plot(fpr,tpr,label="XGB - Tuned - Step 7, auc on test set="+str(auc))
plt.legend(loc=4)
plt.show()


# xgboost - tuned_8 - Fine Tune Regularization Parapebers
xgb_model = xgb.XGBClassifier(objective="binary:logistic", nthread=4, scale_pos_weight=1,random_state=42)

param_dist = {'max_depth': [2],
              'learning_rate': [0.1],
              'subsample': [0.95],
              'n_estimators':[1000],
              'min_child_weight': [4],
              "gamma": [0.1],
              "colsample_bytree": [0.7],
              'reg_alpha':[0.05, 0.1, 0.20, 0.50, 0.75]} 


grid_search = GridSearchCV(xgb_model, 
                                 param_grid= param_dist, 
                                 cv=5,
                                 scoring="roc_auc")


grid_search.fit(X_train_scaled, y_train)

# Look at details


# Create a variable for the best model
best_xgb_tuned = grid_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  grid_search.best_params_)

# Generate predictions with the best model
y_pred_xgb_tuned = best_xgb_tuned.predict(X_test_scaled)

# Confusion Matrix

conf_matrix_xgb_tuned = metrics.confusion_matrix(y_test, y_pred_xgb_tuned)
conf_matrix_xgb_tuned

print(metrics.classification_report(y_test, y_pred_xgb_tuned))


y_pred_proba_xgb_tuned = best_xgb_tuned.predict_proba(X_test_scaled)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_xgb_tuned)
auc = metrics.roc_auc_score(y_test, y_pred_proba_xgb_tuned)
plt.plot(fpr,tpr,label="XGB - Tuned - Step 8, auc on test set="+str(auc))
plt.legend(loc=4)
plt.show()


# xgboost - tuned_9 - Finer Tune Regularization Parapebers
xgb_model = xgb.XGBClassifier(objective="binary:logistic", nthread=4, scale_pos_weight=1,random_state=42)

param_dist = {'max_depth': [2],
              'learning_rate': [0.1],
              'subsample': [0.95],
              'n_estimators':[1000],
              'min_child_weight': [4],
              "gamma": [0.1],
              "colsample_bytree": [0.7],
              'reg_alpha':[0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]} 


grid_search = GridSearchCV(xgb_model, 
                                 param_grid= param_dist, 
                                 cv=5,
                                 scoring="roc_auc")


grid_search.fit(X_train_scaled, y_train)

# Look at details


# Create a variable for the best model
best_xgb_tuned = grid_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  grid_search.best_params_)

# Generate predictions with the best model
y_pred_xgb_tuned = best_xgb_tuned.predict(X_test_scaled)

# Confusion Matrix

conf_matrix_xgb_tuned = metrics.confusion_matrix(y_test, y_pred_xgb_tuned)
conf_matrix_xgb_tuned

print(metrics.classification_report(y_test, y_pred_xgb_tuned))


y_pred_proba_xgb_tuned = best_xgb_tuned.predict_proba(X_test_scaled)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_xgb_tuned)
auc = metrics.roc_auc_score(y_test, y_pred_proba_xgb_tuned)
plt.plot(fpr,tpr,label="XGB - Tuned - Step 9, auc on test set="+str(auc))
plt.legend(loc=4)
plt.show()


# xgboost - tuned_10 - Adjust Learning Rate
xgb_model = xgb.XGBClassifier(objective="binary:logistic", nthread=4, scale_pos_weight=1,random_state=42)

param_dist = {'max_depth': [2],
              'learning_rate': [0.001, 0.01, 0.1],
              'subsample': [0.95],
              'n_estimators':[1000],
              'min_child_weight': [4],
              "gamma": [0.1],
              "colsample_bytree": [0.7],
              'reg_alpha':[0.08]} 


grid_search = GridSearchCV(xgb_model, 
                                 param_grid= param_dist, 
                                 cv=5,
                                 scoring="roc_auc")


grid_search.fit(X_train_scaled, y_train)

# Look at details


# Create a variable for the best model
best_xgb_tuned = grid_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  grid_search.best_params_)

# Generate predictions with the best model
y_pred_xgb_tuned = best_xgb_tuned.predict(X_test_scaled)

# Confusion Matrix

conf_matrix_xgb_tuned = metrics.confusion_matrix(y_test, y_pred_xgb_tuned)
conf_matrix_xgb_tuned

print(metrics.classification_report(y_test, y_pred_xgb_tuned))


y_pred_proba_xgb_tuned = best_xgb_tuned.predict_proba(X_test_scaled)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_xgb_tuned)
auc = metrics.roc_auc_score(y_test, y_pred_proba_xgb_tuned)
plt.plot(fpr,tpr,label="XGB - Tuned - Step 10, auc on test set="+str(auc))
plt.legend(loc=4)
plt.show()


# xgboost - tuned_11 - Adjust Learning Rate
xgb_model = xgb.XGBClassifier(objective="binary:logistic", nthread=4, scale_pos_weight=1,random_state=42)

param_dist = {'max_depth': [2],
              'learning_rate': [0.05, 0.1, 0.15],
              'subsample': [0.95],
              'n_estimators':[1000],
              'min_child_weight': [4],
              "gamma": [0.1],
              "colsample_bytree": [0.7],
              'reg_alpha':[0.08]} 


grid_search = GridSearchCV(xgb_model, 
                                 param_grid= param_dist, 
                                 cv=5,
                                 scoring="roc_auc")


grid_search.fit(X_train_scaled, y_train)

# Look at details


# Create a variable for the best model
best_xgb_tuned = grid_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  grid_search.best_params_)

# Generate predictions with the best model
y_pred_xgb_tuned = best_xgb_tuned.predict(X_test_scaled)

# Confusion Matrix

conf_matrix_xgb_tuned = metrics.confusion_matrix(y_test, y_pred_xgb_tuned)
conf_matrix_xgb_tuned

print(metrics.classification_report(y_test, y_pred_xgb_tuned))


y_pred_proba_xgb_tuned = best_xgb_tuned.predict_proba(X_test_scaled)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_xgb_tuned)
auc = metrics.roc_auc_score(y_test, y_pred_proba_xgb_tuned)
plt.plot(fpr,tpr,label="XGB - Tuned - Step 11, auc on test set="+str(auc))
plt.legend(loc=4)
plt.show()


# xgboost - tuned_12 - Adjust Number of Trees
xgb_model = xgb.XGBClassifier(objective="binary:logistic", nthread=4, scale_pos_weight=1,random_state=42)

param_dist = {'max_depth': [2],
              'learning_rate': [0.1],
              'subsample': [0.95],
              'n_estimators':[1000, 3000, 5000, 7000, 10000],
              'min_child_weight': [4],
              "gamma": [0.1],
              "colsample_bytree": [0.7],
              'reg_alpha':[0.08]} 


grid_search = GridSearchCV(xgb_model, 
                                 param_grid= param_dist, 
                                 cv=5,
                                 scoring="roc_auc")


grid_search.fit(X_train_scaled, y_train)

# Look at details


# Create a variable for the best model
best_xgb_tuned = grid_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  grid_search.best_params_)

# Generate predictions with the best model
y_pred_xgb_tuned = best_xgb_tuned.predict(X_test_scaled)

# Confusion Matrix

conf_matrix_xgb_tuned = metrics.confusion_matrix(y_test, y_pred_xgb_tuned)
conf_matrix_xgb_tuned

print(metrics.classification_report(y_test, y_pred_xgb_tuned))


y_pred_proba_xgb_tuned = best_xgb_tuned.predict_proba(X_test_scaled)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_xgb_tuned)
auc = metrics.roc_auc_score(y_test, y_pred_proba_xgb_tuned)
plt.plot(fpr,tpr,label="XGB - Tuned - Step 12, auc on test set="+str(auc))
plt.legend(loc=4)
plt.show()


# xgboost - tuned_13 - Adjust Number of Trees
xgb_model = xgb.XGBClassifier(objective="binary:logistic", nthread=4, scale_pos_weight=1,random_state=42)

param_dist = {'max_depth': [2],
              'learning_rate': [0.1],
              'subsample': [0.95],
              'n_estimators':[1300, 1400, 1500, 1600, 1700],
              'min_child_weight': [4],
              "gamma": [0.1],
              "colsample_bytree": [0.7],
              'reg_alpha':[0.08]} 


grid_search = GridSearchCV(xgb_model, 
                                 param_grid= param_dist, 
                                 cv=5,
                                 scoring="roc_auc")


grid_search.fit(X_train_scaled, y_train)

# Look at details


# Create a variable for the best model
best_xgb_tuned = grid_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  grid_search.best_params_)

# Generate predictions with the best model
y_pred_xgb_tuned = best_xgb_tuned.predict(X_test_scaled)

# Confusion Matrix

conf_matrix_xgb_tuned = metrics.confusion_matrix(y_test, y_pred_xgb_tuned)
conf_matrix_xgb_tuned

print(metrics.classification_report(y_test, y_pred_xgb_tuned))


y_pred_proba_xgb_tuned = best_xgb_tuned.predict_proba(X_test_scaled)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_xgb_tuned)
auc = metrics.roc_auc_score(y_test, y_pred_proba_xgb_tuned)
plt.plot(fpr,tpr,label="XGB - Tuned - Step 13, auc on test set="+str(auc))
plt.legend(loc=4)
plt.show()


###########3

xgb_final = xgb.XGBClassifier(
 learning_rate =0.1,
 n_estimators=1500,
 max_depth=2,
 min_child_weight=4,
 gamma=0.1,
 subsample=0.95,
 colsample_bytree=0.7,
 reg_alpha=0.08,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 random_state=42)


xgb_final.fit(X_train_scaled,y_train)


y_pred_xgb = xgb_final.predict(X_test_scaled)

# Confusion Matrix
confmtx_xgb = ConfusionMatrix(xgb_final)
confmtx_xgb.fit(X_train_scaled, y_train)
confmtx_xgb.score(X_test_scaled, y_test)


print(metrics.classification_report(y_test, y_pred_xgb))

y_pred_proba_xgb = xgb_final.predict_proba(X_test_scaled)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_xgb)
auc = metrics.roc_auc_score(y_test, y_pred_proba_xgb)
plt.plot(fpr,tpr,label="xgboost, auc on Test set="+str(auc))
plt.legend(loc=4)
plt.show()


## Mode Agnostic Feature importanc
result = permutation_importance(xgb_final,X_test_scaled,y_test,n_repeats=10,random_state=42)

feature_importance = pd.DataFrame({'Feature': X_train_scaled.columns,
                                   'Importance': result.importances_mean,
                                   'Standard Deviation': result.importances_std})
feature_importance = feature_importance.sort_values('Importance', ascending=True)


ax = feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6), yerr='Standard Deviation', capsize=4)
ax.set_xlabel('Permutation Importance')
ax.set_title('XGBoost_tuned - Permutation Importance with Standard Deviation')



## Create a submission file

submissiony_pred_proba_xgb_tuned = xgb_final.predict_proba(submissionX_scaled)[::,1]

kaggle_submission_id = pd.DataFrame(submission_id)
kaggle_submission_prob = pd.DataFrame(submissiony_pred_proba_xgb_tuned)
kaggle_submission = pd.concat([kaggle_submission_id,kaggle_submission_prob],axis=1)
kaggle_submission.rename(columns={kaggle_submission.columns[1]:"Exited"},inplace=True)

kaggle_submission.to_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/submissions/xgb_final.csv",sep=',',index=False)


