import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix,brier_score_loss



df = pd.read_csv('CVD\data\model_ready_data.csv')

# Split the dataset into features and target
X = df.drop(columns=['target'])
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the objective function to optimize
def objective(trial):
    C = trial.suggest_loguniform('C', 1e-5, 1e5)  # Regularization parameter
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])  # Penalty term

    # Define the Logistic Regression model with the suggested parameters
    clf = LogisticRegression(C=C, penalty=penalty, solver='liblinear', random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]  # Get the probability of the positive class (i.e. the "1" class)
    
    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    auc=roc_auc_score(y_test, y_pred_proba)
    brier=brier_score_loss(y_test, y_pred_proba)
    
    
    # Return the negative accuracy (as Optuna minimizes the objective)
    return accuracy

# Create a study object and optimize the objective function
study = optuna.create_study(directions=['maximize'])
study.optimize(objective, n_trials=100)

# Print the results
print('Best trials:', study.best_trials)
print('Best value:', -study.best_value)
print('Best parameters:', study.best_params)

# Train a Logistic Regression model with the best hyperparameters
best_params = study.best_params
clf = LogisticRegression(C=best_params['C'], penalty=best_params['penalty'], solver='liblinear', random_state=42)
clf.fit(X_train, y_train)

# Calculate accuracy on the test set
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]  # Get the probability of the positive class (i.e. the "1" class)
accuracy = accuracy_score(y_test, y_pred)
auc=roc_auc_score(y_test, y_pred_proba)
brier=brier_score_loss(y_test, y_pred_proba)
precision=precision_score(y_test, y_pred)
recall=recall_score(y_test, y_pred)
f1=f1_score(y_test, y_pred)

print('Accuracy of the model with best hyperparameters:', accuracy)
print('AUC of the model with best hyperparameters:', auc)
print('Brier of the model with best hyperparameters:', brier)
print('Precision of the model with best hyperparameters:', precision)
print('Recall of the model with best hyperparameters:', recall)
print('F1 of the model with best hyperparameters:', f1)






