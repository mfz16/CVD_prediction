from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import shap
import pandas as pd
import lightgbm as lgb
import numpy as np

df=pd.read_csv(r"CVD\data\model_ready_data.csv")
X,y=df.iloc[:,:-1],df.iloc[:,-1]


# Step 1: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Define the models
base_models = {
    'Logistic Regression': LogisticRegression(),
    #'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(n_neighbors=9),
    'XGBoost': xgb.XGBClassifier(),
    #'LightGBM': lgb.LGBMClassifier(device='gpu',gpu_platform_id = 1,gpu_device_id = 0)
}

# Step 3: Define hyperparameter grids for each model
param_grids = {
    'Logistic Regression': {'C': [0.01, .1, 1], 'penalty': [None,'l2']},
    #'Decision Tree': {'max_depth': [3, 5, None], 'min_samples_split': [2, 5, 10]},
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
    'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']},
    'KNN': {'n_neighbors': [3, 5, 10], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree']},
    'XGBoost': {'max_depth': [3, 5, 10], 'learning_rate': [0.1, 0.5, 1], 'n_estimators': [50, 100, 200]},
    #'Extra Trees': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
    #'LightGBM': {'num_leaves': [31, 63, 127], 'learning_rate': [0.1, 0.5, 1],}
}

# Train base models with GridSearchCV and collect predictions
base_models_predictions_train = {}
for model_name, model in base_models.items():
    model.fit(X_train, y_train)
    base_models_predictions_train[model_name] = model.predict(X_train)

# Create training data for the meta-learner (stacking)
meta_X_train = np.column_stack([base_models_predictions_train[model_name] for model_name in base_models])

# Step 3: Train the meta-learner (logistic classifier)
meta_learner = LogisticRegression()
meta_learner.fit(meta_X_train, y_train)

# Collect predictions from base models on the test set
base_models_predictions_test = {}
for model_name, model in base_models.items():
    base_models_predictions_test[model_name] = model.predict(X_test)

# Create test data for the meta-learner (stacking)
meta_X_test = np.column_stack([base_models_predictions_test[model_name] for model_name in base_models])

# Step 4: Evaluate the stacking ensemble on the test set
stacking_accuracy = meta_learner.score(meta_X_test, y_test)
print("Stacking Ensemble Test Set Accuracy:", stacking_accuracy)
stacking_auc = cross_val_score(meta_learner, meta_X_test, y_test, cv=5, scoring='roc_auc').mean()
print("Stacking Ensemble Test Set AUC:", stacking_auc)
