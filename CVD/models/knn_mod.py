import numpy as np
import optuna
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.metrics

# Load the data
df = pd.read_csv('CVD/data/model_ready_data.csv')

# Define the objective function
def objective(trial):
    target = df['target']
    data = df.drop(columns=['target'])
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.2, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    valid_x_scaled = scaler.transform(valid_x)

    # Parameters to optimize
    n_neighbors = trial.suggest_int('n_neighbors', 1, 20)

    # KNN model
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(train_x_scaled, train_y)

    # Calculate accuracy
    accuracy = clf.score(valid_x_scaled, valid_y)
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

    # Scale the data
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)

    # Train a model with the best parameters found by Optuna
    best_n_neighbors = trial.params['n_neighbors']
    best_model = KNeighborsClassifier(n_neighbors=best_n_neighbors)
    best_model.fit(train_features_scaled, train_target)

    # Predict on the test set
    test_preds = best_model.predict(test_features_scaled)
    test_probs = best_model.predict_proba(test_features_scaled)[:, 1]
    # Calculate accuracy
    accuracy = sklearn.metrics.accuracy_score(test_target, test_preds)
    print("Test Accuracy:", accuracy)
    auc=sklearn.metrics.roc_auc_score(test_target, test_probs)
    print("Test AUC:", auc)
