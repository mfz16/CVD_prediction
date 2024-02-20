import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

# Load the data
df = pd.read_csv(r"CVD\data\model_ready_data.csv")
X, y = df.iloc[:, :-1], df.iloc[:, -1]

# Step 1: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define XGBoost classifier with parameters
params_xgb = {
    'lambda': 1.0817364168672892e-08,
    'alpha': 0.009867878843687923,
    'max_depth': 8,
    'eta': 0.5877679957668103,
    'gamma': 1.760953662749008e-08,
    'colsample_bytree': 0.9961309271621921,
    'subsample': 0.9989876882231564
}
xgb_clf = xgb.XGBClassifier(**params_xgb)

# Define LightGBM classifier with parameters
params_lgb = {
    'lambda_l1': 0.00019098505955105058,
    'lambda_l2': 9.53567850660335,
    'num_leaves': 145,
    'feature_fraction': 0.5581797344677695,
    'bagging_fraction': 0.9919929967126307,
    'bagging_freq': 1,
    'min_child_samples': 97
}
lgb_clf = lgb.LGBMClassifier(**params_lgb)

# Step 2: Define the base models
base_models = [
    ('Logistic Regression', LogisticRegression(C=5.748838517300915, penalty='l2', solver='liblinear')),
    ('Random Forest', RandomForestClassifier(n_estimators=733, max_depth=21, min_samples_split=6, min_samples_leaf=2)),
    ('SVM', SVC(C=11.442541798976706, gamma=0.2724868129859982)),
    ('KNN', KNeighborsClassifier(n_neighbors=5)),
    ('LightGBM', lgb_clf),
    ('XGBoost', xgb_clf)
]

# Step 3: Define the objective function for optimizing logistic regression hyperparameters
def objective(trial):
    C = trial.suggest_loguniform('C', 1e-5, 1e5)
    lr = LogisticRegression(C=C)
    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=lr
    )
    stacking_clf.fit(X_train, y_train)
    y_prob = stacking_clf.predict_proba(X_test)
    stacking_auc = roc_auc_score(y_test, y_prob[:, 1])
    return stacking_auc

# Step 4: Optimize logistic regression hyperparameters using Optuna
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    # Print the best trial
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Use the best hyperparameters to create the logistic regression model
    best_C = trial.params['C']
    best_lr = LogisticRegression(C=best_C)

    # Define the stacking classifier with the best logistic regression model
    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=best_lr
    )

    # Step 5: Train the stacking classifier
    stacking_clf.fit(X_train, y_train)

    # Predict probabilities on the test set
    y_prob = stacking_clf.predict_proba(X_test)

    # Calculate accuracy and Brier score
    stacking_auc = roc_auc_score(y_test, y_prob[:, 1])
    stacking_brier = brier_score_loss(y_test, y_prob[:, 1])

    print("Stacking Ensemble Test Set AUC:", stacking_auc)
    print("Stacking Ensemble Test Set Brier Score:", stacking_brier)
