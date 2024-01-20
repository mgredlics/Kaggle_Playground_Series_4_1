#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 09:52:45 2024

Binary Classification with Bank Churn Dataset
Kaggle Playground Series - Season 4, Episode 1

@author: mgredlics
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


### Model Building
# Read in File
df = pd.read_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/train.csv")
# Build X and y
feature_cols = ['CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']
X = df[feature_cols]
y = df.Exited

# Split into Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25,random_state=42)

### Set the one hot encoding
categories = [('Gender',['Male','Female']),
              ('Geography',['France','Spain','Germany']),
              ('HasCrCard',[0,1]),
              ('IsActiveMember',[0,1])]

ohe_columns = [x[0] for x in categories]
ohe_categories = [x[1] for x in categories]
enc = OneHotEncoder(sparse_output=False,categories=ohe_categories)

# Create the dataset
X_train_ohe = pd.DataFrame(
    enc.fit_transform(X_train[ohe_columns]),
    columns = enc.get_feature_names_out(),
    index=X_train.index)

# Drop the original columns
X_train_ohe = pd.concat([X_train.drop(ohe_columns,axis=1),X_train_ohe],axis=1)

# Drop the unnecessary Columns
X_train_ohe = X_train_ohe.drop(columns=["Gender_Male","Geography_France","HasCrCard_0.0","IsActiveMember_0.0"])

X_train_ohe

## Apply the Transform to Test Set
#NOTICE ON TEST IT IS TRANSFORM (NOT FIT_TRANSFORM)
X_test_ohe = pd.DataFrame(
    enc.transform(X_test[ohe_columns]),
    columns = enc.get_feature_names_out(),
    index = X_test.index)

# Drop the original columns
X_test_ohe = pd.concat([X_test.drop(ohe_columns,axis=1),X_test_ohe],axis=1)

# Drop the unnecessary Columns
X_test_ohe = X_test_ohe.drop(columns=["Gender_Male","Geography_France","HasCrCard_0.0","IsActiveMember_0.0"])

X_test_ohe

## Set a variable for Balance = 0

## If Balance is 0, create a variable. Still keep Balance
X_train_ohe['Balance_0'] = np.where((X_train_ohe.Balance == 0),1,0)
X_test_ohe['Balance_0'] = np.where((X_test_ohe.Balance == 0),1,0)

## Make custom bins for NumOfProducts
X_train_ohe['NumOfProducts_2'] = np.where((X_train_ohe.NumOfProducts == 2),1,0)
X_train_ohe['NumOfProducts_3more'] = np.where((X_train_ohe.NumOfProducts >= 3),1,0)

X_test_ohe['NumOfProducts_2'] = np.where((X_test_ohe.NumOfProducts == 2),1,0)
X_test_ohe['NumOfProducts_3more'] = np.where((X_test_ohe.NumOfProducts >= 3),1,0)

X_train_ohe = X_train_ohe.drop("NumOfProducts",axis=1)
X_test_ohe = X_test_ohe.drop("NumOfProducts",axis=1)

### Min Max Scaler
scaler = MinMaxScaler()
numeric_columns = ['CreditScore','Age','Tenure','Balance','EstimatedSalary']

## Get Scale information on Train
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_ohe[numeric_columns]),
    columns = scaler.get_feature_names_out(),
    index=X_train_ohe.index)

X_train_scaled = pd.concat([X_train_ohe.drop(numeric_columns,axis=1),X_train_scaled],axis=1)

## Apply to Test
#NOTICE ON TEST IT IS TRANSFORM (NOT FIT_TRANSFORM)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_ohe[numeric_columns]),
    columns = scaler.get_feature_names_out(),
    index=X_test_ohe.index)

X_test_scaled = pd.concat([X_test_ohe.drop(numeric_columns,axis=1),X_test_scaled],axis=1)

## Save for Later

X_train_scaled.to_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/PreProcessedData/X_train_scaled.csv",sep=',',index=True)
X_test_scaled.to_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/PreProcessedData/X_test_scaled.csv",sep=',',index=True)
y_train.to_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/PreProcessedData/y_train.csv",sep=',',index=True)
y_test.to_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/PreProcessedData/y_test.csv",sep=',',index=True)


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

submission_id.to_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/PreProcessedData/submission_id.csv",sep=',',index=True)
submissionX_scaled.to_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/PreProcessedData/submissionX_scaled.csv",sep=',',index=True)
