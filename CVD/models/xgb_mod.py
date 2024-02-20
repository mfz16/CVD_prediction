import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import sklearn.metrics
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv('CVD/data/model_ready_data.csv')

# Define the objective function
def objective(trial):
    target = df['target']
    data = df.drop(columns=['target'])
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(train_x, label=train_y)
    dvalid = xgb.DMatrix(valid_x, label=valid_y)

    param = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "verbosity": 0,
        "booster": "gbtree",
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "eta": trial.suggest_float("eta", 1e-8, 1.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
    }

    bst = xgb.train(param, dtrain)
    preds = bst.predict(dvalid)
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
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        **trial.params
    }
    dtrain = xgb.DMatrix(train_features, label=train_target)
    bst = xgb.train(xgb_params, dtrain)

    # Predict on the test set
    dtest = xgb.DMatrix(test_features)
    test_preds = bst.predict(dtest)
    test_preds = bst.predict(dtest)

    # Calculate accuracy
    test_pred_labels = np.round(test_preds)
    accuracy = sklearn.metrics.accuracy_score(test_target, test_pred_labels)
    print("Test Accuracy:", accuracy)

    # Calculate AUC
    auc = sklearn.metrics.roc_auc_score(test_target, test_preds)
    print("Test AUC:", auc)
