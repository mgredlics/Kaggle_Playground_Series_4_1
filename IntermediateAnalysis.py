"""
Created on Tue Jan 16 12:58:45 2024

Binary Classification with Bank Churn Dataset
Kaggle Playground Series - Season 4, Episode 1

@author: mgredlics
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


## Surname
# Surname has 2,726 values
len(pd.unique(X["Surname"]))
# Top 10 are:
# Hsia, T'ien, Kao, Hs?, TS'ui, Maclean, P'eng, H?, Hsueh, Shih
X['Surname'].value_counts().nlargest(10)    

## CreditScore
# No big outliers hereQ
fig, ax = subplots(figsize=(8,8))
X.hist('CreditScore',bins=10,color='blue',ax=ax)
ax.set_ylabel('Count')

## Geography
# 3 Different Geographies France, Spain, Germany
len(pd.unique(X["Geography"]))
X['Geography'].value_counts().nlargest(3)    

# Not sure how to get the barplot

## Gender
# 2 Different Genders Male/Female
len(pd.unique(X["Gender"]))
X['Gender'].value_counts().nlargest(2)  

## Age
# No big outliers hereQ
fig, ax = subplots(figsize=(8,8))
X.hist('Age',bins=10,color='blue',ax=ax)
ax.set_ylabel('Count')

## Tenure
# No big outliers hereQ
fig, ax = subplots(figsize=(8,8))
X.hist('Tenure',bins=10,color='blue',ax=ax)
ax.set_ylabel('Count')

## Balance
# A lot of zeros
fig, ax = subplots(figsize=(8,8))
X.hist('Balance',bins=10,color='blue',ax=ax)
ax.set_ylabel('Count')

## NumOfProducts
# Mainly 1 or 2. Perhaps look at 1 and 2+
fig, ax = subplots(figsize=(8,8))
X.hist('NumOfProducts',bins=10,color='blue',ax=ax)
ax.set_ylabel('Count')

## HasCrCard
# Just 0/1
len(pd.unique(X["HasCrCard"]))
X['HasCrCard'].value_counts().nlargest(2)

## IsActiveMember
# Just 0/1
len(pd.unique(X["IsActiveMember"]))
X['IsActiveMember'].value_counts().nlargest(2)

## EstimatedSalary
# Not too bad on the outliers
fig, ax = subplots(figsize=(8,8))
X.hist('EstimatedSalary',bins=10,color='blue',ax=ax)
ax.set_ylabel('Count')

## Set the easy 0/1 dummy variables
categories = [('Gender',['Male','Female']),
              ('Geography',['France','Spain','Germany']),
              ('HasCrCard',[0,1]),
              ('IsActiveMember',[0,1])]

ohe_columns = [x[0] for x in categories]
ohe_categories = [x[1] for x in categories]
enc = OneHotEncoder(sparse_output=False,categories=ohe_categories)

X_ohe = pd.DataFrame(
    enc.fit_transform(X[ohe_columns]),
    columns = enc.get_feature_names_out(),
    index=X.index)

X_ohe = pd.concat([X.drop(ohe_columns,axis=1),X_ohe],axis=1)
X_ohe


## Drop Surname (Explore later?)

X_ohe_minus = X_ohe.drop('Surname',axis=1)


## If Balance is 0, create a variable
X_ohe_minus['Balance_0'] = np.where((X_ohe_minus.Balance == 0),1,0)


## Make bins for NumOf Products
X_ohe_minus['NumOfProducts_1'] = np.where((X_ohe_minus.NumOfProducts == 1),1,0)
X_ohe_minus['NumOfProducts_2'] = np.where((X_ohe_minus.NumOfProducts == 2),1,0)
X_ohe_minus['NumOfProducts_3more'] = np.where((X_ohe_minus.NumOfProducts >= 3),1,0)

X_ohe_minus = X_ohe_minus.drop("NumOfProducts",axis=1)


print(X_ohe_minus["CreditScore"].describe())
print(X_ohe_minus["Age"].describe())
print(X_ohe_minus["Balance"].describe())



# Min Max Scaler
scaler = MinMaxScaler()
numeric_columns = ['CreditScore','Age','Tenure','Balance','EstimatedSalary']

X_scaled = pd.DataFrame(
    scaler.fit_transform(X_ohe_minus[numeric_columns]),
    columns = scaler.get_feature_names_out(),
    index=X_ohe_minus.index)
    scaler.fit_transform(X_train_ohe_minus[numeric_columns]),
    columns = scaler.get_feature_names_out(),
    index=X_train_ohe_minus.index)

X_scaled = pd.concat([X_ohe_minus.drop(numeric_columns,axis=1),X_scaled],axis=1)


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

rand_search.fit(X_scaled, y)

# Create a variable for the best model
best_xgb1 = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)


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

submissiony_pred_proba_xgb1_tuned = best_xgb1.predict_proba(submissionX_scaled)[::,1]

kaggle_submission_id = pd.DataFrame(submission_id)
kaggle_submission_prob = pd.DataFrame(submissiony_pred_proba_xgb1_tuned)
kaggle_submission = pd.concat([kaggle_submission_id,kaggle_submission_prob],axis=1)
kaggle_submission.rename(columns={kaggle_submission.columns[1]:"Exited"},inplace=True)

#'gamma': 0.2887799119200372, 'learning_rate': 0.02536913319536114, 'max_depth': 3, 'min_child_weight': 5, 'n_estimators': 1771, 'subsample': 0.694026611835673}
kaggle_submission.to_csv("/home/mgredlics/Python/Kaggle/Playground_Series_Season4_Episode1/submission_xgb_tuned_v5_wholedf.csv",sep=',',index=False)
