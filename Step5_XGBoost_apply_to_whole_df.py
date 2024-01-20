#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 15:52:54 2024

@author: mgredlics
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import xgboost as xgb
from yellowbrick.classifier import ConfusionMatrix
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

### Model Building
# Read in File
df = pd.read_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/train.csv")
# Build X and y
feature_cols = ['CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']
X = df[feature_cols]
y = df.Exited

### Set the one hot encoding
categories = [('Gender',['Male','Female']),
              ('Geography',['France','Spain','Germany']),
              ('HasCrCard',[0,1]),
              ('IsActiveMember',[0,1])]

ohe_columns = [x[0] for x in categories]
ohe_categories = [x[1] for x in categories]
enc = OneHotEncoder(sparse_output=False,categories=ohe_categories)

# Create the dataset
X_ohe = pd.DataFrame(
    enc.fit_transform(X[ohe_columns]),
    columns = enc.get_feature_names_out(),
    index=X.index)

# Drop the original columns
X_ohe = pd.concat([X.drop(ohe_columns,axis=1),X_ohe],axis=1)

# Drop the unnecessary Columns
X_ohe = X_ohe.drop(columns=["Gender_Male","Geography_France","HasCrCard_0.0","IsActiveMember_0.0"])

X_ohe

## Set a variable for Balance = 0

## If Balance is 0, create a variable. Still keep Balance
X_ohe['Balance_0'] = np.where((X_ohe.Balance == 0),1,0)


## Make custom bins for NumOfProducts
X_ohe['NumOfProducts_2'] = np.where((X_ohe.NumOfProducts == 2),1,0)
X_ohe['NumOfProducts_3more'] = np.where((X_ohe.NumOfProducts >= 3),1,0)

X_ohe = X_ohe.drop("NumOfProducts",axis=1)


### Min Max Scaler
scaler = MinMaxScaler()
numeric_columns = ['CreditScore','Age','Tenure','Balance','EstimatedSalary']

## Get Scale information on Train
X_scaled = pd.DataFrame(
    scaler.fit_transform(X_ohe[numeric_columns]),
    columns = scaler.get_feature_names_out(),
    index=X_ohe.index)

X_scaled = pd.concat([X_ohe.drop(numeric_columns,axis=1),X_scaled],axis=1)


### Apply to Submission
submissionX = pd.read_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/test.csv")
# Build X and y
submission_id = submissionX.id

submissionX = submissionX[feature_cols]

submissionX_ohe = pd.DataFrame(
    enc.transform(submissionX[ohe_columns]),
    columns = enc.get_feature_names_out(),
    index = submissionX.index)

# Drop the original columns
submissionX_ohe = pd.concat([submissionX.drop(ohe_columns,axis=1),submissionX_ohe],axis=1)

# Drop the unnecessary Columns
submissionX_ohe = submissionX_ohe.drop(columns=["Gender_Male","Geography_France","HasCrCard_0.0","IsActiveMember_0.0"])

## If Balance is 0, create a variable. Still keep Balance
submissionX_ohe['Balance_0'] = np.where((submissionX_ohe.Balance == 0),1,0)

## Make custom bins for NumOfProducts
submissionX_ohe['NumOfProducts_2'] = np.where((submissionX_ohe.NumOfProducts == 2),1,0)
submissionX_ohe['NumOfProducts_3more'] = np.where((submissionX_ohe.NumOfProducts >= 3),1,0)

submissionX_ohe = submissionX_ohe.drop("NumOfProducts",axis=1)

## Min Max Scalser

submissionX_scaled = pd.DataFrame(
    scaler.transform(submissionX_ohe[numeric_columns]),
    columns = scaler.get_feature_names_out(),
    index=submissionX_ohe.index)

submissionX_scaled = pd.concat([submissionX_ohe.drop(numeric_columns,axis=1),submissionX_scaled],axis=1)


xgb_wholedf_final = xgb.XGBClassifier(
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

xgb_wholedf_final.fit(X_scaled,y)


y_pred_xgb = xgb_wholedf_final.predict(X_scaled)

# Confusion Matrix
confmtx_xgb = ConfusionMatrix(xgb_wholedf_final)
confmtx_xgb.fit(X_scaled, y)
confmtx_xgb.score(X_scaled, y)


print(metrics.classification_report(y, y_pred_xgb))

y_pred_proba_xgb = xgb_wholedf_final.predict_proba(X_scaled)[::,1]
fpr, tpr, _ = metrics.roc_curve(y,  y_pred_proba_xgb)
auc = metrics.roc_auc_score(y, y_pred_proba_xgb)
plt.plot(fpr,tpr,label="xgboost, auc on Entire set="+str(auc))
plt.legend(loc=4)
plt.show()


## Mode Agnostic Feature importanc
result = permutation_importance(xgb_wholedf_final,X_scaled,y,n_repeats=10,random_state=42)

feature_importance = pd.DataFrame({'Feature': X_scaled.columns,
                                   'Importance': result.importances_mean,
                                   'Standard Deviation': result.importances_std})
feature_importance = feature_importance.sort_values('Importance', ascending=True)


ax = feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6), yerr='Standard Deviation', capsize=4)
ax.set_xlabel('Permutation Importance')
ax.set_title('XGBoost_tuned - Permutation Importance with Standard Deviation')



## Create a submission file

submissiony_pred_proba_xgb_tuned = xgb_wholedf_final.predict_proba(submissionX_scaled)[::,1]

kaggle_submission_id = pd.DataFrame(submission_id)
kaggle_submission_prob = pd.DataFrame(submissiony_pred_proba_xgb_tuned)
kaggle_submission = pd.concat([kaggle_submission_id,kaggle_submission_prob],axis=1)
kaggle_submission.rename(columns={kaggle_submission.columns[1]:"Exited"},inplace=True)

kaggle_submission.to_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/submissions/xgb_whole_df_final.csv",sep=',',index=False)


