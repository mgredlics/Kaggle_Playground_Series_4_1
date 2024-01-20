#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 10:46:33 2024

Binary Classification with Bank Churn Dataset
Kaggle Playground Series - Season 4, Episode 1

@author: mgredlics
"""

import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
from yellowbrick.classifier import ConfusionMatrix
from sklearn.model_selection import RandomizedSearchCV
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


# xgboost
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

xgb_model.fit(X_train_scaled,y_train)


y_pred_xgb = xgb_model.predict(X_test_scaled)

# Confusion Matrix
confmtx_xgb = ConfusionMatrix(xgb_model)
confmtx_xgb.fit(X_train_scaled, y_train)
confmtx_xgb.score(X_test_scaled, y_test)


print(metrics.classification_report(y_test, y_pred_xgb))

y_pred_proba_xgb = xgb_model.predict_proba(X_test_scaled)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_xgb)
auc = metrics.roc_auc_score(y_test, y_pred_proba_xgb)
plt.plot(fpr,tpr,label="xgboost, auc on Test set="+str(auc))
plt.legend(loc=4)
plt.show()



## Fit Models
# xgboost - tuned_random
xgb_model = xgb.XGBClassifier(objective="binary:logistic", nthread=4, scale_pos_weight=1,random_state=42)

param_dist = {'max_depth': randint(2, 10),
              'learning_rate': uniform(0.01, 0.3),
              'subsample': uniform(0.05, 1),
              'n_estimators':randint(50, 2000),
              'min_child_weight': randint(3,15),
              "gamma": uniform(0, 0.5)} 


rand_search = RandomizedSearchCV(xgb_model, 
                                 param_distributions = param_dist, 
                                 n_iter=20, 
                                 cv=5,
                                 scoring="roc_auc")


rand_search.fit(X_train_scaled, y_train)

# Create a variable for the best model
best_xgb_tuned = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)

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

## Mode Agnostic Feature importanc
result = permutation_importance(best_xgb_tuned,X_test_scaled,y_test,n_repeats=10,random_state=42)

feature_importance = pd.DataFrame({'Feature': X_train_scaled.columns,
                                   'Importance': result.importances_mean,
                                   'Standard Deviation': result.importances_std})
feature_importance = feature_importance.sort_values('Importance', ascending=True)


ax = feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6), yerr='Standard Deviation', capsize=4)
ax.set_xlabel('Permutation Importance')
ax.set_title('XGBoost_tuned - Permutation Importance with Standard Deviation')






## Create a submission file

submissiony_pred_proba_xgb_tuned = best_xgb_tuned.predict_proba(submissionX_scaled)[::,1]

kaggle_submission_id = pd.DataFrame(submission_id)
kaggle_submission_prob = pd.DataFrame(submissiony_pred_proba_xgb_tuned)
kaggle_submission = pd.concat([kaggle_submission_id,kaggle_submission_prob],axis=1)
kaggle_submission.rename(columns={kaggle_submission.columns[1]:"Exited"},inplace=True)

kaggle_submission.to_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/submissions/xgb_tuned.csv",sep=',',index=False)

