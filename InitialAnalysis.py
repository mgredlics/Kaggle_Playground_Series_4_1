#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binary Classification with Bank Churn Dataset
Kaggle Playground Series - Season 4, Episode 1
"""

import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn import metrics
from scipy.stats import randint,uniform
import xgboost as xgb
from catboost import CatBoost,CatBoostClassifier, Pool
from CatBoostClassifier import randomized_search


# Read in file
churn = pd.read_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/train.csv")
churn.columns


# Remove 'id', 'CustomerID' as should not ever need those
churn = churn.drop(columns=['id','CustomerId'])

# Split into features and target variables
feature_cols = ['Surname','CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']
X = churn[feature_cols]
y = churn.Exited

np.mean(y) # .21 Dataset is not balanced

# Split into Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25,random_state=42)

### Investigate Variables
## Whole dataset
X_train.info()
# No missing

## Surname
# Surname has 2,726 values
len(pd.unique(X_train["Surname"]))
# Top 10 are:
# Hsia, T'ien, Kao, Hs?, TS'ui, Maclean, P'eng, H?, Hsueh, Shih
X_train['Surname'].value_counts().nlargest(10)    

## CreditScore
# No big outliers hereQ
fig, ax = subplots(figsize=(8,8))
X_train.hist('CreditScore',bins=10,color='blue',ax=ax)
ax.set_ylabel('Count')

## Geography
# 3 Different Geographies France, Spain, Germany
len(pd.unique(X_train["Geography"]))
X_train['Geography'].value_counts().nlargest(3)    

# Not sure how to get the barplot

## Gender
# 2 Different Genders Male/Female
len(pd.unique(X_train["Gender"]))
X_train['Gender'].value_counts().nlargest(2)  

## Age
# No big outliers hereQ
fig, ax = subplots(figsize=(8,8))
X_train.hist('Age',bins=10,color='blue',ax=ax)
ax.set_ylabel('Count')

## Tenure
# No big outliers hereQ
fig, ax = subplots(figsize=(8,8))
X_train.hist('Tenure',bins=10,color='blue',ax=ax)
ax.set_ylabel('Count')

## Balance
# A lot of zeros
fig, ax = subplots(figsize=(8,8))
X_train.hist('Balance',bins=10,color='blue',ax=ax)
ax.set_ylabel('Count')

## NumOfProducts
# Mainly 1 or 2. Perhaps look at 1 and 2+
fig, ax = subplots(figsize=(8,8))
X_train.hist('NumOfProducts',bins=10,color='blue',ax=ax)
ax.set_ylabel('Count')

## HasCrCard
# Just 0/1
len(pd.unique(X_train["HasCrCard"]))
X_train['HasCrCard'].value_counts().nlargest(2)

## IsActiveMember
# Just 0/1
len(pd.unique(X_train["IsActiveMember"]))
X_train['IsActiveMember'].value_counts().nlargest(2)

## EstimatedSalary
# Not too bad on the outliers
fig, ax = subplots(figsize=(8,8))
X_train.hist('EstimatedSalary',bins=10,color='blue',ax=ax)
ax.set_ylabel('Count')

## Set the easy 0/1 dummy variables
categories = [('Gender',['Male','Female']),
              ('Geography',['France','Spain','Germany']),
              ('HasCrCard',[0,1]),
              ('IsActiveMember',[0,1])]

ohe_columns = [x[0] for x in categories]
ohe_categories = [x[1] for x in categories]
enc = OneHotEncoder(sparse_output=False,categories=ohe_categories)

X_train_ohe = pd.DataFrame(
    enc.fit_transform(X_train[ohe_columns]),
    columns = enc.get_feature_names_out(),
    index=X_train.index)

X_train_ohe = pd.concat([X_train.drop(ohe_columns,axis=1),X_train_ohe],axis=1)
X_train_ohe

#NOTICE ON TEST IT IS TRANSFORM (NOT FIT_TRANSFORM)
X_test_ohe = pd.DataFrame(
    enc.transform(X_test[ohe_columns]),
    columns = enc.get_feature_names_out(),
    index = X_test.index)

X_test_ohe = pd.concat([X_test.drop(ohe_columns,axis=1),X_test_ohe],axis=1)
X_test_ohe

## Drop Surname (Explore later?)

X_train_ohe_minus = X_train_ohe.drop('Surname',axis=1)
X_test_ohe_minus = X_test_ohe.drop('Surname',axis=1)

## If Balance is 0, create a variable
X_train_ohe_minus['Balance_0'] = np.where((X_train_ohe_minus.Balance == 0),1,0)
X_test_ohe_minus['Balance_0'] = np.where((X_test_ohe_minus.Balance == 0),1,0)

## Make bins for NumOf Products
X_train_ohe_minus['NumOfProducts_1'] = np.where((X_train_ohe_minus.NumOfProducts == 1),1,0)
X_train_ohe_minus['NumOfProducts_2'] = np.where((X_train_ohe_minus.NumOfProducts == 2),1,0)
X_train_ohe_minus['NumOfProducts_3more'] = np.where((X_train_ohe_minus.NumOfProducts >= 3),1,0)

X_test_ohe_minus['NumOfProducts_1'] = np.where((X_test_ohe_minus.NumOfProducts == 1),1,0)
X_test_ohe_minus['NumOfProducts_2'] = np.where((X_test_ohe_minus.NumOfProducts == 2),1,0)
X_test_ohe_minus['NumOfProducts_3more'] = np.where((X_test_ohe_minus.NumOfProducts >= 3),1,0)

X_train_ohe_minus = X_train_ohe_minus.drop("NumOfProducts",axis=1)
X_test_ohe_minus = X_test_ohe_minus.drop("NumOfProducts",axis=1)

print(X_train_ohe_minus["CreditScore"].describe())
print(X_train_ohe_minus["Age"].describe())
print(X_train_ohe_minus["Balance"].describe())


print(X_test_ohe_minus["CreditScore"].describe())
print(X_test_ohe_minus["Age"].describe())
print(X_test_ohe_minus["Balance"].describe())

# Min Max Scaler
scaler = MinMaxScaler()
numeric_columns = ['CreditScore','Age','Tenure','Balance','EstimatedSalary']

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_ohe_minus[numeric_columns]),
    columns = scaler.get_feature_names_out(),
    index=X_train_ohe_minus.index)

X_train_scaled = pd.concat([X_train_ohe_minus.drop(numeric_columns,axis=1),X_train_scaled],axis=1)

#NOTICE ON TEST IT IS TRANSFORM (NOT FIT_TRANSFORM)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_ohe_minus[numeric_columns]),
    columns = scaler.get_feature_names_out(),
    index=X_test_ohe_minus.index)

X_test_scaled = pd.concat([X_test_ohe_minus.drop(numeric_columns,axis=1),X_test_scaled],axis=1)

## Fit Models
# Logistic Regression
logreg = LogisticRegression(random_state=16)

logreg.fit(X_train_scaled,y_train)

y_pred_logreg = logreg.predict(X_test_scaled)

conf_matrix_logreg = metrics.confusion_matrix(y_test, y_pred_logreg)
conf_matrix_logreg

print(metrics.classification_report(y_test, y_pred_logreg))


y_pred_proba = logreg.predict_proba(X_test_scaled)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="LogReg, auc="+str(auc))
plt.legend(loc=4)
plt.show()

## Apply to Submission - Prepare file
submissionX = pd.read_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/test.csv")
submission_id = submissionX.id

submissionX = submissionX.drop(columns=['id','CustomerId'])

submissionX_ohe = pd.DataFrame(
    enc.transform(submissionX[ohe_columns]),
    columns = enc.get_feature_names_out(),
    index = submissionX.index)

submissionX_ohe = pd.concat([submissionX.drop(ohe_columns,axis=1),submissionX_ohe],axis=1)
submissionX_ohe

submissionX_ohe_minus = submissionX_ohe.drop('Surname',axis=1)

submissionX_ohe_minus['Balance_0'] = np.where((submissionX_ohe_minus.Balance == 0),1,0)
submissionX_ohe_minus['NumOfProducts_1'] = np.where((submissionX_ohe_minus.NumOfProducts == 1),1,0)
submissionX_ohe_minus['NumOfProducts_2'] = np.where((submissionX_ohe_minus.NumOfProducts == 2),1,0)
submissionX_ohe_minus['NumOfProducts_3more'] = np.where((submissionX_ohe_minus.NumOfProducts >= 3),1,0)

submissionX_ohe_minus = submissionX_ohe_minus.drop("NumOfProducts",axis=1)

submissionX_scaled = pd.DataFrame(
    scaler.transform(submissionX_ohe_minus[numeric_columns]),
    columns = scaler.get_feature_names_out(),
    index=submissionX_ohe_minus.index)

submissionX_scaled = pd.concat([submissionX_ohe_minus.drop(numeric_columns,axis=1),submissionX_scaled],axis=1)

submissiony_pred_proba_logreg = logreg.predict_proba(submissionX_scaled)[::,1]

kaggle_submission_id = pd.DataFrame(submission_id)
kaggle_submission_prob = pd.DataFrame(submissiony_pred_proba_logreg)
kaggle_submission = pd.concat([kaggle_submission_id,kaggle_submission_prob],axis=1)
kaggle_submission.rename(columns={kaggle_submission.columns[1]:"Exited"},inplace=True)

kaggle_submission.to_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/submission1.csv",sep=',',index=False)



## Fit Models
# Decision Tree
dectree = DecisionTreeClassifier()

dectree.fit(X_train_scaled,y_train)


y_pred_tree = dectree.predict(X_test_scaled)

conf_matrix_tree = metrics.confusion_matrix(y_test, y_pred_tree)
conf_matrix_tree

print(metrics.classification_report(y_test, y_pred_tree))


y_pred_proba_tree = dectree.predict_proba(X_test_scaled)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_tree)
auc = metrics.roc_auc_score(y_test, y_pred_proba_tree)
plt.plot(fpr,tpr,label="Decision_Tree, auc="+str(auc))
plt.legend(loc=4)
plt.show()


## Fit Models
# Random Forest
rf = RandomForestClassifier()

rf.fit(X_train_scaled,y_train)


y_pred_rf = rf.predict(X_test_scaled)

conf_matrix_rf = metrics.confusion_matrix(y_test, y_pred_rf)
conf_matrix_rf

print(metrics.classification_report(y_test, y_pred_rf))


y_pred_proba_rf = rf.predict_proba(X_test_scaled)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_rf)
auc = metrics.roc_auc_score(y_test, y_pred_proba_rf)
plt.plot(fpr,tpr,label="RandomForest, auc="+str(auc))
plt.legend(loc=4)
plt.show()


## Apply to Submission - Prepare file
submissionX = pd.read_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/test.csv")
submission_id = submissionX.id

submissionX = submissionX.drop(columns=['id','CustomerId'])

submissionX_ohe = pd.DataFrame(
    enc.transform(submissionX[ohe_columns]),
    columns = enc.get_feature_names_out(),
    index = submissionX.index)

submissionX_ohe = pd.concat([submissionX.drop(ohe_columns,axis=1),submissionX_ohe],axis=1)
submissionX_ohe

submissionX_ohe_minus = submissionX_ohe.drop('Surname',axis=1)

submissionX_ohe_minus['Balance_0'] = np.where((submissionX_ohe_minus.Balance == 0),1,0)
submissionX_ohe_minus['NumOfProducts_1'] = np.where((submissionX_ohe_minus.NumOfProducts == 1),1,0)
submissionX_ohe_minus['NumOfProducts_2'] = np.where((submissionX_ohe_minus.NumOfProducts == 2),1,0)
submissionX_ohe_minus['NumOfProducts_3more'] = np.where((submissionX_ohe_minus.NumOfProducts >= 3),1,0)

submissionX_ohe_minus = submissionX_ohe_minus.drop("NumOfProducts",axis=1)

submissionX_scaled = pd.DataFrame(
    scaler.transform(submissionX_ohe_minus[numeric_columns]),
    columns = scaler.get_feature_names_out(),
    index=submissionX_ohe_minus.index)

submissionX_scaled = pd.concat([submissionX_ohe_minus.drop(numeric_columns,axis=1),submissionX_scaled],axis=1)

submissiony_pred_proba_rf = rf.predict_proba(submissionX_scaled)[::,1]

kaggle_submission_id = pd.DataFrame(submission_id)
kaggle_submission_prob = pd.DataFrame(submissiony_pred_proba_rf)
kaggle_submission = pd.concat([kaggle_submission_id,kaggle_submission_prob],axis=1)
kaggle_submission.rename(columns={kaggle_submission.columns[1]:"Exited"},inplace=True)

kaggle_submission.to_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/submission_rf.csv",sep=',',index=False)

## Fit Models
# Random Forest - Tuned
param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

rand_search = RandomizedSearchCV(rf, 
                                 param_distributions = param_dist, 
                                 n_iter=5, 
                                 cv=5,
                                 random_state=42,
                                 verbose=1,
                                 n_jobs=1,
                                 return_train_score=True)

rand_search.fit(X_train_scaled, y_train)

# Create a variable for the best model
best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)

# Generate predictions with the best model
y_pred_rf_tuned = best_rf.predict(X_test_scaled)

conf_matrix_rf_tuned = metrics.confusion_matrix(y_test, y_pred_rf_tuned)
conf_matrix_rf_tuned

print(metrics.classification_report(y_test, y_pred_rf_tuned))


y_pred_proba_rf_tuned = best_rf.predict_proba(X_test_scaled)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_rf_tuned)
auc = metrics.roc_auc_score(y_test, y_pred_proba_rf_tuned)
plt.plot(fpr,tpr,label="RandomForest - Tuned, auc="+str(auc))
plt.legend(loc=4)
plt.show()


## Apply to Submission - Prepare file
submissionX = pd.read_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/test.csv")
submission_id = submissionX.id

submissionX = submissionX.drop(columns=['id','CustomerId'])

submissionX_ohe = pd.DataFrame(
    enc.transform(submissionX[ohe_columns]),
    columns = enc.get_feature_names_out(),
    index = submissionX.index)

submissionX_ohe = pd.concat([submissionX.drop(ohe_columns,axis=1),submissionX_ohe],axis=1)
submissionX_ohe

submissionX_ohe_minus = submissionX_ohe.drop('Surname',axis=1)

submissionX_ohe_minus['Balance_0'] = np.where((submissionX_ohe_minus.Balance == 0),1,0)
submissionX_ohe_minus['NumOfProducts_1'] = np.where((submissionX_ohe_minus.NumOfProducts == 1),1,0)
submissionX_ohe_minus['NumOfProducts_2'] = np.where((submissionX_ohe_minus.NumOfProducts == 2),1,0)
submissionX_ohe_minus['NumOfProducts_3more'] = np.where((submissionX_ohe_minus.NumOfProducts >= 3),1,0)

submissionX_ohe_minus = submissionX_ohe_minus.drop("NumOfProducts",axis=1)

submissionX_scaled = pd.DataFrame(
    scaler.transform(submissionX_ohe_minus[numeric_columns]),
    columns = scaler.get_feature_names_out(),
    index=submissionX_ohe_minus.index)

submissionX_scaled = pd.concat([submissionX_ohe_minus.drop(numeric_columns,axis=1),submissionX_scaled],axis=1)

submissiony_pred_proba_rf_tuned = best_rf.predict_proba(submissionX_scaled)[::,1]

kaggle_submission_id = pd.DataFrame(submission_id)
kaggle_submission_prob = pd.DataFrame(submissiony_pred_proba_rf_tuned)
kaggle_submission = pd.concat([kaggle_submission_id,kaggle_submission_prob],axis=1)
kaggle_submission.rename(columns={kaggle_submission.columns[1]:"Exited"},inplace=True)

kaggle_submission.to_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/submission_rf_tuned.csv",sep=',',index=False)

## Fit Models
# xgboost
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

xgb_model.fit(X_train_scaled,y_train)


y_pred_xgb = xgb_model.predict(X_test_scaled)

conf_matrix_xgb = metrics.confusion_matrix(y_test, y_pred_xgb)
conf_matrix_xgb

print(metrics.classification_report(y_test, y_pred_xgb))


y_pred_proba_xgb = xgb.predict_proba(X_test_scaled)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_xgb)
auc = metrics.roc_auc_score(y_test, y_pred_proba_xgb)
plt.plot(fpr,tpr,label="xgboost, auc="+str(auc))
plt.legend(loc=4)
plt.show()



## Fit Models
# xgboost - tuned_1
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)


param_dist = {'max_depth': randint(2, 10),
              'learning_rate': uniform(0.01, 0.3),
              'subsample': uniform(0.05, 1),
              'n_estimators':randint(50, 2000),
              'min_child_weight': randint(3,15),
              "gamma": uniform(0, 0.5)} 

rand_search = RandomizedSearchCV(xgb_model, 
                                 param_distributions = param_dist, 
                                 n_iter=500, 
                                 cv=10,
                                 scoring="roc_auc")

rand_search.fit(X_train_scaled, y_train)

# Create a variable for the best model
best_xgb1 = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)

# Generate predictions with the best model
y_pred_xgb1_tuned = best_xgb1.predict(X_test_scaled)

conf_matrix_xgb1_tuned = metrics.confusion_matrix(y_test, y_pred_xgb1_tuned)
conf_matrix_xgb1_tuned

print(metrics.classification_report(y_test, y_pred_xgb1_tuned))


y_pred_proba_xgb1_tuned = best_xgb1.predict_proba(X_test_scaled)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_xgb1_tuned)
auc = metrics.roc_auc_score(y_test, y_pred_proba_xgb1_tuned)
plt.plot(fpr,tpr,label="XGB1 - Tuned, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# xgboost - tuned_2
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)


param_dist = {'max_depth':  [3],
              'learning_rate': [0.041531528772958785],
              'subsample': [0.5906100155154264],
              'n_estimators': [1170],
              'min_child_weight': [6],
              'gamma': [0.22469900037842366]}

rand_search = RandomizedSearchCV(xgb_model, 
                                 param_distributions = param_dist, 
                                 n_iter=20, 
                                 cv=5,
                                 scoring="roc_auc")

rand_search.fit(X_train_scaled, y_train)

# Create a variable for the best model
best_xgb2 = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)

# Generate predictions with the best model
y_pred_xgb2_tuned = best_xgb2.predict(X_test_scaled)

conf_matrix_xgb2_tuned = metrics.confusion_matrix(y_test, y_pred_xgb2_tuned)
conf_matrix_xgb2_tuned

print(metrics.classification_report(y_test, y_pred_xgb2_tuned))


y_pred_proba_xgb2_tuned = best_xgb2.predict_proba(X_test_scaled)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_xgb2_tuned)
auc = metrics.roc_auc_score(y_test, y_pred_proba_xgb2_tuned)
plt.plot(fpr,tpr,label="XGB2 - Tuned, auc="+str(auc))
plt.legend(loc=4)
plt.show()


## Apply to Submission - Prepare file
submissionX = pd.read_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/test.csv")
submission_id = submissionX.id

submissionX = submissionX.drop(columns=['id','CustomerId'])

submissionX_ohe = pd.DataFrame(
    enc.transform(submissionX[ohe_columns]),
    columns = enc.get_feature_names_out(),
    index = submissionX.index)

submissionX_ohe = pd.concat([submissionX.drop(ohe_columns,axis=1),submissionX_ohe],axis=1)
submissionX_ohe

submissionX_ohe_minus = submissionX_ohe.drop('Surname',axis=1)

submissionX_ohe_minus['Balance_0'] = np.where((submissionX_ohe_minus.Balance == 0),1,0)
submissionX_ohe_minus['NumOfProducts_1'] = np.where((submissionX_ohe_minus.NumOfProducts == 1),1,0)
submissionX_ohe_minus['NumOfProducts_2'] = np.where((submissionX_ohe_minus.NumOfProducts == 2),1,0)
submissionX_ohe_minus['NumOfProducts_3more'] = np.where((submissionX_ohe_minus.NumOfProducts >= 3),1,0)

submissionX_ohe_minus = submissionX_ohe_minus.drop("NumOfProducts",axis=1)

submissionX_scaled = pd.DataFrame(
    scaler.transform(submissionX_ohe_minus[numeric_columns]),
    columns = scaler.get_feature_names_out(),
    index=submissionX_ohe_minus.index)

submissionX_scaled = pd.concat([submissionX_ohe_minus.drop(numeric_columns,axis=1),submissionX_scaled],axis=1)

submissiony_pred_proba_xgb2_tuned = best_xgb2.predict_proba(submissionX_scaled)[::,1]

kaggle_submission_id = pd.DataFrame(submission_id)
kaggle_submission_prob = pd.DataFrame(submissiony_pred_proba_xgb2_tuned)
kaggle_submission = pd.concat([kaggle_submission_id,kaggle_submission_prob],axis=1)
kaggle_submission.rename(columns={kaggle_submission.columns[1]:"Exited"},inplace=True)

#'max_depth':  [3],'learning_rate': [0.053840993108296306],'subsample': [0.7133360275112508],'n_estimators': [645],'min_child_weight': [3],
#.88576 on kaggle
#kaggle_submission.to_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/submission_xgb2_tuned.csv",sep=',',index=False)


#'max_depth':  [3],'learning_rate': [0.03398507907964156],'subsample': [0.8642243718041335],'n_estimators': [769],'min_child_weight': [6],
#.88597 on kaggle
#kaggle_submission.to_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/submission_xgb2_tuned_v2.csv",sep=',',index=False)


#param_dist = {'max_depth':  [3],'learning_rate': [0.051615549684644595],'subsample': [0.8068475651780523],'n_estimators': [774],'min_child_weight': [6],'gamma': [0.031511678053833814]}
#.88593 on kaggle
#kaggle_submission.to_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/submission_xgb2_tuned_v3.csv",sep=',',index=False)

#param_dist = {'max_depth':  [3],'learning_rate': [0.041531528772958785],'subsample': [0.5906100155154264],'n_estimators': [1170],'min_child_weight': [6],'gamma': [0.22469900037842366]}
#.88593 on kaggle
#kaggle_submission.to_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/submission_xgb2_tuned_v4.csv",sep=',',index=False)



#### Next Try 'Catboost'


## Fit Models
# catboost - tuned_1
catboost_model = CatBoostClassifier(eval_metric='AUC')


param_dist = {'depth': randint(2, 10),
              'learning_rate': uniform(0.01, 0.3),
              'iterations': randint(10,100)} 

rand_search = catboost_model.randomized_search(param_distributions = param_dist,
                                               X = X_train_scaled,
                                               y = y_train,
                                               cv=5,
                                               n_iter=100,
                                               partition_random_seed = 42)

# Create a variable for the best model
best_catboost = rand_search['params']


# Print the best hyperparameters
print('Best hyperparameters:',  rand_search['params'])

# Generate predictions with the best model
y_pred_catboost_tuned = pd.DataFrame(catboost_model.predict_proba(X_test_scaled))
y_pred_catboost_tuned = y_pred_catboost_tuned[y_pred_catboost_tuned.columns[1]]



y_pred_proba_catboost_tuned = catboost_model.predict_proba(X_test_scaled)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_catboost_tuned)
auc = metrics.roc_auc_score(y_test, y_pred_proba_catboost_tuned)
plt.plot(fpr,tpr,label="Catboost - Tuned, auc="+str(auc))
plt.legend(loc=4)
plt.show()


## Apply to Submission - Prepare file
submissionX = pd.read_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/test.csv")
submission_id = submissionX.id

submissionX = submissionX.drop(columns=['id','CustomerId'])

submissionX_ohe = pd.DataFrame(
    enc.transform(submissionX[ohe_columns]),
    columns = enc.get_feature_names_out(),
    index = submissionX.index)

submissionX_ohe = pd.concat([submissionX.drop(ohe_columns,axis=1),submissionX_ohe],axis=1)
submissionX_ohe

submissionX_ohe_minus = submissionX_ohe.drop('Surname',axis=1)

submissionX_ohe_minus['Balance_0'] = np.where((submissionX_ohe_minus.Balance == 0),1,0)
submissionX_ohe_minus['NumOfProducts_1'] = np.where((submissionX_ohe_minus.NumOfProducts == 1),1,0)
submissionX_ohe_minus['NumOfProducts_2'] = np.where((submissionX_ohe_minus.NumOfProducts == 2),1,0)
submissionX_ohe_minus['NumOfProducts_3more'] = np.where((submissionX_ohe_minus.NumOfProducts >= 3),1,0)

submissionX_ohe_minus = submissionX_ohe_minus.drop("NumOfProducts",axis=1)

submissionX_scaled = pd.DataFrame(
    scaler.transform(submissionX_ohe_minus[numeric_columns]),
    columns = scaler.get_feature_names_out(),
    index=submissionX_ohe_minus.index)

submissionX_scaled = pd.concat([submissionX_ohe_minus.drop(numeric_columns,axis=1),submissionX_scaled],axis=1)

submissiony_pred_proba_catboost_tuned = catboost_model.predict_proba(submissionX_scaled)[::,1]

kaggle_submission_id = pd.DataFrame(submission_id)
kaggle_submission_prob = pd.DataFrame(submissiony_pred_proba_catboost_tuned)
kaggle_submission = pd.concat([kaggle_submission_id,kaggle_submission_prob],axis=1)
kaggle_submission.rename(columns={kaggle_submission.columns[1]:"Exited"},inplace=True)

#param_dist = Best hyperparameters: {'depth': 4.0, 'learning_rate': 0.24777705946292303, 'iterations': 64.0}
#.88464 on kaggle
kaggle_submission.to_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/submission_catboost_tuned_v1.csv",sep=',',index=False)

