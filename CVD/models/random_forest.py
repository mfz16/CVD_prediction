import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics

# Load the data
df = pd.read_csv('CVD/data/model_ready_data.csv')

# Define the objective function
def objective(trial):
    target = df['target']
    data = df.drop(columns=['target'])
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.2, random_state=42)

    # Parameters to optimize
    n_estimators = trial.suggest_int('n_estimators', 10, 1000)
    max_depth = trial.suggest_int('max_depth', 2, 32)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)

    # Random Forest model
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    clf.fit(train_x, train_y)

    # Calculate accuracy
    accuracy = clf.score(valid_x, valid_y)
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
    best_n_estimators = trial.params['n_estimators']
    best_max_depth = trial.params['max_depth']
    best_min_samples_split = trial.params['min_samples_split']
    best_min_samples_leaf = trial.params['min_samples_leaf']

    best_model = RandomForestClassifier(
        n_estimators=best_n_estimators,
        max_depth=best_max_depth,
        min_samples_split=best_min_samples_split,
        min_samples_leaf=best_min_samples_leaf,
        random_state=42
    )
    best_model.fit(train_features, train_target)

    # Predict on the test set
    test_preds = best_model.predict(test_features)

    # Calculate accuracy
    accuracy = sklearn.metrics.accuracy_score(test_target, test_preds)
    print("Test Accuracy:", accuracy)
