from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle as pk

df = pd.read_csv(r"CVD\data\model_ready_data.csv")
X, y = df.iloc[:, :-1], df.iloc[:, -1]

# Step 1: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
params_xgb={
    'lambda':1.0817364168672892e-08,
    'alpha':0.009867878843687923,
    'max_depth':8,
    'eta':0.5877679957668103,
    'gamma':1.760953662749008e-08,
    'colsample_bytree': 0.9961309271621921,
    'subsample':0.9989876882231564}
xgb_clf = xgb.XGBClassifier(**params_xgb)
# Step 2: Define the base models
base_models = [
    ('Logistic Regression', LogisticRegression(C=5.748838517300915, penalty='l2', solver='liblinear')),
    ('Random Forest', RandomForestClassifier(  n_estimators= 733,
    max_depth=21,
    min_samples_split=6,
    min_samples_leaf=2)),
    ('SVM', SVC(C=11.442541798976706,gamma= 0.2724868129859982)),
    ('KNN', KNeighborsClassifier(n_neighbors=5)),
    #('Decision Tree', DecisionTreeClassifier()),
    #('LightGBM', lgb.LGBMClassifier(device='gpu',gpu_platform_id = 1,gpu_device_id = 0)),
    ('LightGBM', lgb.LGBMClassifier( lambda_l1= 0.00019098505955105058,
    lambda_l2=9.53567850660335,
    num_leaves= 145,
    feature_fraction= 0.5581797344677695,
    bagging_fraction=0.9919929967126307,
    bagging_freq=1,
    min_child_samples= 97)),
    ('XGBoost', xgb_clf)
]

# Step 3: Define the stacking classifier
stacking_clf = StackingClassifier(
    estimators=base_models,
    #final_estimator=LogisticRegression()
    final_estimator=LogisticRegression(C= 94173.77608767607, penalty='l1', solver='liblinear')
)

# Step 4: Train the stacking classifier
stacking_clf.fit(X_train, y_train)
with open('CVD\stacking_model.pkl', 'wb') as f:
    pk.dump(stacking_clf, f)
yp=stacking_clf.predict(X_test)
y_prob=stacking_clf.predict_proba(X_test)
# Step 5: Evaluate the stacking classifier on the test set
stacking_accuracy = stacking_clf.score(X_test, y_test)
print("Stacking Ensemble Test Set Accuracy:", stacking_accuracy)
stacking_auc = roc_auc_score(y_test, y_prob[:, 1]) #pobaility of class 1
print(stacking_clf.classes_)
print("Stacking Ensemble Test Set AUC:", stacking_auc)
stacking_brier=brier_score_loss(y_test, y_prob[:, 1])
print("Stacking Ensemble Test Set Brier:", stacking_brier)