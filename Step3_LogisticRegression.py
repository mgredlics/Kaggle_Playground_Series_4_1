#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 10:26:31 2024

Binary Classification with Bank Churn Dataset
Kaggle Playground Series - Season 4, Episode 1

@author: mgredlics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from yellowbrick.classifier import ConfusionMatrix
from sklearn.linear_model import LogisticRegression


### Model Building
# Read in Preprocessed Files

X_train_scaled = pd.read_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/PreProcessedData/X_train_scaled.csv",index_col=0)
X_test_scaled = pd.read_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/PreProcessedData/X_test_scaled.csv", index_col=0)
y_train = pd.read_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/PreProcessedData/y_train.csv", index_col=0)
y_test = pd.read_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/PreProcessedData/y_test.csv", index_col=0)

submissionX_scaled = pd.read_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/PreProcessedData/submissionX_scaled.csv", index_col=0)
submission_id = pd.read_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/PreProcessedData/submission_id.csv", index_col=0)


## Logistic Regression
logreg = LogisticRegression(random_state=16)
logreg.fit(X_train_scaled,y_train)

y_pred_logreg = logreg.predict(X_test_scaled)

# Confusion Matrix
confmtx_logreg = ConfusionMatrix(logreg)
confmtx_logreg.fit(X_train_scaled, y_train)
confmtx_logreg.score(X_test_scaled, y_test)

# Stats
classification_logreg = (classification_report(y_test, y_pred_logreg))
print(classification_logreg)

# Get Predictions for Probabilities
y_pred_proba = logreg.predict_proba(X_test_scaled)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="LogReg, auc on Test set="+str(auc))
plt.legend(loc=4)
plt.show()

# Get Feature importance

coefficients = logreg.coef_[0]
feature_importance = pd.DataFrame({'Feature': X_train_scaled.columns, 'Importance': np.abs(coefficients)})
feature_importance = feature_importance.sort_values('Importance', ascending=True)
feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))


## Create a submission file

submissiony_pred_proba_logreg = logreg.predict_proba(submissionX_scaled)[::,1]

kaggle_submission_id = pd.DataFrame(submission_id)
kaggle_submission_prob = pd.DataFrame(submissiony_pred_proba_logreg)
kaggle_submission = pd.concat([kaggle_submission_id,kaggle_submission_prob],axis=1)
kaggle_submission.rename(columns={kaggle_submission.columns[1]:"Exited"},inplace=True)

kaggle_submission.to_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/submissions/submission_logR_clean.csv",sep=',',index=False)

