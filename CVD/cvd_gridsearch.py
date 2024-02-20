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

df=pd.read_csv(r"CVD\data\model_ready_data_without_outliers.csv")
X,y=df.iloc[:,:-1],df.iloc[:,-1]


# Step 1: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Define the models
models = {
    'Logistic Regression': LogisticRegression(),
    #'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    #'KNN': KNeighborsClassifier(),
    'XGBoost': xgb.XGBClassifier(),
    #'LightGBM': lgb.LGBMClassifier(device='gpu',gpu_platform_id = 1,gpu_device_id = 0)
}

# Step 3: Define hyperparameter grids for each model
param_grids = {
    'Logistic Regression': {'C': [0.01, .1, 1], 'penalty': [None,'l2']},
    #'Decision Tree': {'max_depth': [3, 5, None], 'min_samples_split': [2, 5, 10]},
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
    'SVM': {'C': [ 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']},
    'KNN': {'n_neighbors': [3, 5, 10], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree']},
    'XGBoost': {'max_depth': [3, 5, 10], 'learning_rate': [0.1, 0.5, 1], 'n_estimators': [50, 100, 200]},
    #'Extra Trees': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
    #'LightGBM': {'num_leaves': [31, 63, 127], 'learning_rate': [0.1, 0.5, 1],}
}

# Perform Grid Search and Cross-Validation for each model
for model_name, model in models.items():
    print(f"\n{model_name}:")

    # Step 4: Perform Grid Search with 5-fold Cross-Validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Display all grid search results
    print("Grid Search Results:")
    for params, mean_score, scores in zip(
        grid_search.cv_results_['params'],
        grid_search.cv_results_['mean_test_score'],
        grid_search.cv_results_['std_test_score']
    ):
        print(f"Hyperparameters: {params}, Mean Accuracy: {mean_score:.4f} (+/- {scores:.4f})")

    # Get the best model from grid search
    best_model = grid_search.best_estimator_

    # Step 5: Evaluate the best model using k-fold Cross-Validation
    # Perform 5-fold cross-validation on the training set
    cv_accuracy = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')

    # Display cross-validation results
    print("\nCross-Validation Results:")
    for fold, accuracy in enumerate(cv_accuracy, start=1):
        print(f"Fold {fold}: Accuracy = {accuracy:.4f}")

    # Step 6: Evaluate the best model on the test set
    # Assess the performance of the best model on the held-out test set
    test_accuracy = best_model.score(X_test, y_test)
    test_auc=roc_auc_score(y_test, best_model.predict_proba(X_test)[:,1])
    print("\nBest model is:",best_model)
    print("\n Best model score:",best_model.score)
    print("\nTest Set Accuracy:", test_accuracy)
    print("\nTest Set AUC:", test_auc)
