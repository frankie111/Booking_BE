

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC



# Function to plot the history of model accuracy
def plot_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.legend()
    plt.show()




def load_data(filepath):
    df = pd.read_csv(filepath)
    df['room'] = df['room'].astype('category').cat.codes

    # Preparing input features.
    X = df[['day_of_week', 'month', 'week_of_year', 'room']]

    # Preparing separate target variables for each time slot.
    y = df[['nineToEleven', 'elevenToOne', 'oneToThree', 'threeToFive']]

    return X, y

def train_multioutput_model(X, y):
    base_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    multi_output_rf = MultiOutputClassifier(base_rf, n_jobs=-1)
    multi_output_rf.fit(X, y)
    return multi_output_rf

# Function to split data into training and testing sets
def split_data(X, y, test_size=0.1, random_state=0):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)




def predict_and_save(models, X_test, y_test, filename='predictions_test.csv'):
    results = pd.DataFrame(y_test)
    for time_slot in models:
        model = models[time_slot]
        results[f'Predicted_{time_slot}'] = model.predict(X_test)
    results.to_csv(filename, index=False)
    print("Predictions saved to", filename)

def build_xgboost_model():
    return XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0)

def evaluate_model(model, X_test, y_test, time_slot):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'{time_slot} Accuracy: {accuracy}')
    print(confusion_matrix(y_test, predictions))
    print(predictions)


# Function to perform k-fold cross-validation
def k_fold_cross_validation(X, y, n_splits=5):
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    fold_no = 1
    for train_index, test_index in skf.split(X, y):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        model = build_xgboost_model()
        model.fit(X_train_fold, y_train_fold)
        predictions = model.predict(X_test_fold)

        accuracy = accuracy_score(y_test_fold, predictions)
        print(f'Fold {fold_no}, Accuracy: {accuracy}')
        fold_no += 1


# Function to build and return the model
def build_model():
    return RandomForestClassifier(random_state=0)


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    # Initialize an accuracy list
    accuracies = []

    # Loop through each interval (column) in y_test
    for i in range(y_test.shape[1]):
        interval_name = y_test.columns[i]
        interval_accuracy = accuracy_score(y_test.iloc[:, i], predictions[:, i])
        accuracies.append(interval_accuracy)
        print(f"Accuracy for {interval_name}: {interval_accuracy:.4f}")

    # Optionally, you could return the accuracies if you want to use them elsewhere
    return accuracies

def perform_grid_search(X_train, y_train):
    # Define a parameter grid to search
    param_grid = {
        'n_estimators': [100, 175, 200],  # Number of trees in the forest
        'learning_rate': [0.01, 0.1],  # Step size shrinkage used to prevent overfitting
        'max_depth': [3, 4, 5, 7],  # Maximum depth of a tree
        'subsample': [0.8, 0.9, 1],  # Subsample ratio of the training instances
        'colsample_bytree': [0.8, 1],  # Subsample ratio of columns when constructing each tree
    }

    # Initialize the XGBClassifier
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0)

    # Setup GridSearchCV
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=8, n_jobs=-1, verbose=1)

    # Perform the grid search on the training data
    grid_search.fit(X_train, y_train)

    # Return the best estimator
    return grid_search.best_estimator_

# def main():
#     # filepath = 'preprocessed_month_categorical.csv'
#     filepath = 'merged_data.csv'
#     X, y = load_data(filepath)
#
#     models = {}
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
#     y_test_results = pd.DataFrame(index=X_test.index)
#
#     for time_slot in y.columns:
#         model = SVC()
#         model.fit(X_train, y_train[time_slot])
#         models[time_slot] = model
#
#         print(f"Evaluating model for {time_slot}:")
#         evaluate_model(model, X_test, y_test[time_slot], time_slot)
#
#         y_test_results[f'Predicted_{time_slot}'] = model.predict(X_test)
#
#     predictions_filename = 'predictions_test.csv'
#     y_test_results.to_csv(predictions_filename, index=False)
#     print(f"Predictions saved to {predictions_filename}")


# def main():
#     filepath = 'merged_data.csv'
#     X, y = load_data(filepath)
#
#     models = {}
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
#
#
#     model = train_multioutput_model(X_train, y_train)
#
#     evaluate_model(model, X_test, y_test)
#     y_test_results = pd.DataFrame(index=X_test.index)
#
#
#     # with open('multioutput_model.pkl', 'wb') as file:
#     #     pickle.dump(model, file)
#     # print("Multi-output model saved to multioutput_model.pkl")
#
#     # for time_slot in y.columns:
#     #     # Perform grid search to find the best model
#     #     best_model = perform_grid_search(X_train, y_train[time_slot])
#     #     models[time_slot] = best_model
#     #
#     #     print(f"Evaluating model for {time_slot}:")
#     #     evaluate_model(best_model, X_test, y_test[time_slot], time_slot)
#     #
#     #     y_test_results[f'Predicted_{time_slot}'] = best_model.predict(X_test)
#
#     predictions_filename = 'predictions_test.csv'
#     y_test_results.to_csv(predictions_filename, index=False)
#     print(f"Predictions saved to {predictions_filename}")
# #
# # if __name__ == '__main__':
#     main()


def main():
    filepath = 'merged_data.csv'
    X, y = load_data(filepath)

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    # Training a multi-output model
    model = train_multioutput_model(X_train, y_train)

    # Saving the trained model to a file
    # joblib.dump(model, 'multioutput_model.pkl')

    with open('multioutput_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    print("Multi-output model saved to multioutput_model.pkl")

    # Making predictions on the test set
    predictions = model.predict(X_test)

    # Evaluating and printing model accuracy
    evaluate_model(model, X_test, y_test)

    # Preparing DataFrame with actual and predicted values
    results = pd.DataFrame(y_test).reset_index(drop=True)
    results_pred = pd.DataFrame(predictions, columns=[f'Predicted_{col}' for col in y_test.columns])
    final_results = pd.concat([results, results_pred], axis=1)

    # Saving predictions to CSV
    final_results.to_csv('predictions_test.csv', index=False)
    print("Predictions and actual values saved to predictions_test.csv")

if __name__ == '__main__':
    main()
