import numpy as np
import optuna
import pandas as pd
import lightgbm as lgb
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split

df=pd.read_csv('CVD\data\model_ready_data.csv')
# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    #data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    target=df['target']
    data=df.drop(columns=['target'])
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.2)
    dtrain = lgb.Dataset(train_x, label=train_y)

    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    gbm = lgb.train(param, dtrain)
    preds = gbm.predict(valid_x)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    
    
    # Split data into train and test
    train_data, test_data = train_test_split(df, test_size=0.25, random_state=42)
    train_target = train_data['target']
    train_features = train_data.drop(columns=['target'])
    test_target = test_data['target']
    test_features = test_data.drop(columns=['target'])

    # Train a model with the best parameters found by Optuna
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        **trial.params
    }
    lgb_train = lgb.Dataset(train_features, train_target)
    lgb_model = lgb.train(lgb_params, lgb_train)

    # Predict on the test set
    test_preds = lgb_model.predict(test_features)

    # Calculate accuracy
    test_pred_labels = np.round(test_preds)
    accuracy = sklearn.metrics.accuracy_score(test_target, test_pred_labels)
    print("Test Accuracy:", accuracy)

    # Calculate AUC
    auc = sklearn.metrics.roc_auc_score(test_target, test_preds)
    print("Test AUC:", auc)