# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import accuracy_score
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# import matplotlib.pyplot as plt
#
#
# def load_data(filepath):
#     data = pd.read_csv(filepath)
#     # Convert date to datetime and extract features if needed
#     data['date'] = pd.to_datetime(data['date'])
#     # Assuming 'availability' is the target variable, which needs to be defined based on your criteria (e.g., if a meeting room is booked in any of the slots, it's not available)
#     return data
#
#
# def preprocess_data(data):
#     # Convert categorical data to numerical if needed, e.g., room names to categorical codes
#     data['room'] = data['room'].astype('category').cat.codes
#     # Here, define your 'y' based on the meeting room availability; this is an example
#     # You might need to adjust based on your actual availability criteria
#     data['availability'] = 1 - data[['nineToEleven', 'elevenToOne', 'oneToThree', 'threeToFive']].max(axis=1)
#     X = data[['capacity', 'day_of_week', 'month', 'week_of_year', 'total_booked_desks_first_half',
#               'total_booked_desks_second_half', 'room']]
#     y = data['availability']
#     return X, y
#
#
# def build_and_train_model(X_train, y_train):
#     # Example model: RandomForestClassifier
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#     return model
#
#
# def evaluate_model(model, X_test, y_test):
#     predictions = model.predict(X_test)
#     accuracy = accuracy_score(y_test, predictions)
#     print(f"Accuracy: {accuracy}")
#
#
# def main():
#     data = load_data('merged_data.csv')
#     X, y = preprocess_data(data)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     model = build_and_train_model(X_train, y_train)
#     evaluate_model(model, X_test, y_test)
#
#     # Feature Importance
#     feature_importances = pd.Series(model.feature_importances_, index=X.columns)
#     feature_importances.nlargest(10).plot(kind='barh')
#     plt.show()
#
#
# if __name__ == '__main__':
#     main()
import pickle

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import accuracy_score
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# import matplotlib.pyplot as plt
#
# def load_data(filepath):
#     data = pd.read_csv(filepath)
#     # Convert date to datetime and extract features if needed
#     data['date'] = pd.to_datetime(data['date'])
#     return data
#
# def preprocess_data(data):
#     # Example of adding a new feature: number of booked intervals per day
#     data['num_booked_intervals'] = data[['nineToEleven', 'elevenToOne', 'oneToThree', 'threeToFive']].sum(axis=1)
#     data['room'] = data['room'].astype('category').cat.codes
#     X = data[['capacity', 'day_of_week', 'month', 'week_of_year',
#               'total_booked_desks_first_half', 'total_booked_desks_second_half',
#               'room', 'num_booked_intervals']]
#     y = data[['nineToEleven', 'elevenToOne', 'oneToThree', 'threeToFive']]
#     return X, y
# def build_and_train_model(X_train, y_train):
#     # Tuning RandomForest parameters example
#     model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
#     model.fit(X_train, y_train)
#     return model
#
# def evaluate_model(model, X_test, y_test):
#     predictions = model.predict(X_test)
#     # Assuming binary 'availability', we calculate accuracy for each interval
#     accuracies = [accuracy_score(y_test.iloc[:, i], predictions[:, i]) for i in range(y_test.shape[1])]
#     for i, acc in enumerate(accuracies):
#         print(f"Interval {y_test.columns[i]} Accuracy: {acc}")
#
# def main():
#     data = load_data('merged_data.csv')
#     X, y = preprocess_data(data)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     model = build_and_train_model(X_train, y_train)
#     evaluate_model(model, X_test, y_test)
#
#     # For RandomForest, showing feature importances
#     feature_importances = pd.Series(model.feature_importances_, index=X.columns)
#     feature_importances.nlargest(10).plot(kind='barh')
#     plt.show()
#
# if __name__ == '__main__':
#     main()


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, KFold
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix
# import matplotlib.pyplot as plt
#
#
# def plot_history(histories, key='accuracy'):
#     plt.figure(figsize=(16, 10))
#     for name, history in histories:
#         val = plt.plot(history.epoch, history.history['val_' + key],
#                        '--', label=name.title() + ' Val')
#         plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
#                  label=name.title() + ' Train')
#     plt.xlabel('Epochs')
#     plt.ylabel(key.replace('_', ' ').title())
#     plt.legend()
#     plt.xlim([0, max(history.epoch)])


# def load_data():
#     df = pd.read_csv('merged_data.csv')
#     # Assuming 'available' is a binary target feature indicating room availability
#     df['available'] = df.apply(
#         lambda row: 1 if row['nineToEleven'] + row['elevenToOne'] + row['oneToThree'] + row['threeToFive'] > 0 else 0,
#         axis=1)
#     return df

# def load_data(filepath):
#     df = pd.read_csv(filepath)
#
#     # Preparing input features.
#     X = df[['capacity', 'day_of_week', 'month', 'week_of_year', 'total_booked_desks_first_half',
#               'total_booked_desks_second_half', 'room']]
#
#     # Preparing separate target variables for each time slot.
#     y = df[['nineToEleven', 'elevenToOne', 'oneToThree', 'threeToFive']]
#
#     return X, y

# def load_data(filepath):
#     df = pd.read_csv(filepath)
#
#     # One-hot encode the 'room' feature
#     df = pd.get_dummies(df, columns=['room'])
#
#     # Selecting relevant features
#     X = df[['capacity', 'month', 'week_of_year', 'total_booked_desks_first_half',
#             'total_booked_desks_second_half'] + [col for col in df.columns if col.startswith('room_')]]
#
#     # Preparing separate target variables for each time slot
#     y = df[['nineToEleven', 'elevenToOne', 'oneToThree', 'threeToFive']]
#
#     return X, y
#
#
#
# def save_predictions_to_csv(original_values, predictions, filename='predictions.csv'):
#     df = pd.DataFrame({'Original Values': original_values, 'Predictions': predictions})
#     df.to_csv(filename, index=False)
#
#
#
# def build_model(model_type='RandomForest'):
#     if model_type == 'RandomForest':
#         return RandomForestClassifier()
#     elif model_type == 'XGBoost':
#         return XGBClassifier(use_label_encoder=False, eval_metric='logloss')
#
#
# def k_fold_cross_validation(X, y, model, n_splits=5):
#     kf = KFold(n_splits=n_splits)
#     scores = []
#
#     for train_index, test_index in kf.split(X):
#         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#         y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#
#         model.fit(X_train, y_train)
#         predictions = model.predict(X_test)
#         score = accuracy_score(y_test, predictions)
#         scores.append(score)
#
#     return np.mean(scores), scores
#
#
# def split_data(X, y):
#     return train_test_split(X, y, test_size=0.2, random_state=42)
#
#
#
#
# # def main():
# #     filepath = 'merged_data.csv'
# #     X, y = load_data(filepath)
# #     # X, y = reshape_data(df)
# #     X_train, X_test, y_train, y_test = split_data(X, y)
# #
# #     model_rf = build_model('RandomForest')
# #     model_xgb = build_model('XGBoost')
# #
# #     rf_score, rf_scores = k_fold_cross_validation(X, y, model_rf)
# #     xgb_score, xgb_scores = k_fold_cross_validation(X, y, model_xgb)
# #
# #     print(f"RandomForest Average Score: {rf_score}")
# #     print(f"XGBoost Average Score: {xgb_score}")
# #
# #     # Train on the whole dataset and make predictions for future use
# #     model_rf.fit(X, y)
# #     model_xgb.fit(X, y)
# #
# #     rf_predictions = model_rf.predict(X)
# #     xgb_predictions = model_xgb.predict(X)
# #
# #     save_predictions_to_csv(y, rf_predictions, 'rf_predictions.csv')
# #     save_predictions_to_csv(y, xgb_predictions, 'xgb_predictions.csv')
#
# def evaluate_model(model, X_test, y_test, time_slot):
#     predictions = model.predict(X_test)
#     accuracy = accuracy_score(y_test, predictions)
#     print(f'{time_slot} Accuracy: {accuracy}')
#     print(confusion_matrix(y_test, predictions))
#
# def main():
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
#     predictions_filename = 'predictions_test2.csv'
#     y_test_results.to_csv(predictions_filename, index=False)
#     print(f"Predictions saved to {predictions_filename}")
#
#
# if __name__ == '__main__':
#     main()

# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix
# import pickle
# import matplotlib.pyplot as plt
# from xgboost import XGBClassifier
# from sklearn.svm import SVC
#
#
#
# # Function to plot the history of model accuracy
# def plot_history(history):
#     plt.figure(figsize=(10, 5))
#     plt.plot(history['train_acc'], label='Train Accuracy')
#     plt.plot(history['val_acc'], label='Validation Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.title('Model Accuracy Over Epochs')
#     plt.legend()
#     plt.show()
#
#
#
#
# def load_data(filepath):
#     df = pd.read_csv(filepath)
#
#     # Preparing input features.
#     X = df[['capacity', 'day_of_week', 'month', 'week_of_year', 'total_booked_desks_first_half',
#               'total_booked_desks_second_half', 'room']]
#
#     # Preparing separate target variables for each time slot.
#     y = df[['nineToEleven', 'elevenToOne', 'oneToThree', 'threeToFive']]
#
#     return X, y
#
#
# # Function to split data into training and testing sets
# def split_data(X, y, test_size=0.1, random_state=0):
#     return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
#
#
#
#
# def predict_and_save(models, X_test, y_test, filename='predictions_test.csv'):
#     results = pd.DataFrame(y_test)
#     for time_slot in models:
#         model = models[time_slot]
#         results[f'Predicted_{time_slot}'] = model.predict(X_test)
#     results.to_csv(filename, index=False)
#     print("Predictions saved to", filename)
#
# def build_xgboost_model():
#     return XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0)
#
# def evaluate_model(model, X_test, y_test, time_slot):
#     predictions = model.predict(X_test)
#     accuracy = accuracy_score(y_test, predictions)
#     print(f'{time_slot} Accuracy: {accuracy}')
#     print(confusion_matrix(y_test, predictions))
#
#
# # Function to perform k-fold cross-validation
# def k_fold_cross_validation(X, y, n_splits=5):
#     skf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
#     fold_no = 1
#     for train_index, test_index in skf.split(X, y):
#         X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
#         y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
#
#         model = build_xgboost_model()
#         model.fit(X_train_fold, y_train_fold)
#         predictions = model.predict(X_test_fold)
#
#         accuracy = accuracy_score(y_test_fold, predictions)
#         print(f'Fold {fold_no}, Accuracy: {accuracy}')
#         fold_no += 1
#
#
# # Function to build and return the model
# def build_model():
#     return RandomForestClassifier(random_state=0)
#
# def perform_grid_search(X_train, y_train):
#     # Define a parameter grid to search
#     param_grid = {
#         'n_estimators': [100, 175, 200],  # Number of trees in the forest
#         'learning_rate': [0.01, 0.1],  # Step size shrinkage used to prevent overfitting
#         'max_depth': [3, 4, 5, 7],  # Maximum depth of a tree
#         'subsample': [0.8, 0.9, 1],  # Subsample ratio of the training instances
#         'colsample_bytree': [0.8, 1],  # Subsample ratio of columns when constructing each tree
#     }
#
#     # Initialize the XGBClassifier
#     xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0)
#
#     # Setup GridSearchCV
#     grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=8, n_jobs=-1, verbose=1)
#
#     # Perform the grid search on the training data
#     grid_search.fit(X_train, y_train)
#
#     # Return the best estimator
#     return grid_search.best_estimator_
#
# def main():
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
#
#
# # def main():
# #     filepath = 'preprocessed_month_categorical.csv'
# #     X, y = load_data(filepath)
# #
# #     models = {}
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
# #     y_test_results = pd.DataFrame(index=X_test.index)
# #
# #     for time_slot in y.columns:
# #         # Perform grid search to find the best model
# #         best_model = perform_grid_search(X_train, y_train[time_slot])
# #         models[time_slot] = best_model
# #
# #         print(f"Evaluating model for {time_slot}:")
# #         evaluate_model(best_model, X_test, y_test[time_slot], time_slot)
# #
# #         y_test_results[f'Predicted_{time_slot}'] = best_model.predict(X_test)
# #
# #     predictions_filename = 'predictions_test.csv'
# #     y_test_results.to_csv(predictions_filename, index=False)
# #     print(f"Predictions saved to {predictions_filename}")
# #
# # if __name__ == '__main__':
# #     main()
#
# if __name__ == '__main__':
#     main()


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def plot_history(histories, key='accuracy'):
    plt.figure(figsize=(16, 10))
    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_' + key],
                       '--', label=name.title() + ' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title() + ' Train')
    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()
    plt.xlim([0, max(history.epoch)])

def load_data():
    df = pd.read_csv('merged_data.csv')
    # Assuming 'available' is a binary target feature indicating room availability
    df['available'] = df.apply(
        lambda row: 1 if row['nineToEleven'] + row['elevenToOne'] + row['oneToThree'] + row['threeToFive'] > 0 else 0,
        axis=1)
    return df

def reshape_data(df):
    # Select relevant features
    X = df[['capacity', 'day_of_week', 'month', 'week_of_year', 'total_booked_desks_first_half',
            'total_booked_desks_second_half']]
    y = df['available']
    return X, y

def build_model(model_type='RandomForest'):
    if model_type == 'RandomForest':
        return RandomForestClassifier()
    elif model_type == 'XGBoost':
        return XGBClassifier(use_label_encoder=False, eval_metric='logloss')

def k_fold_cross_validation(X, y, model, n_splits=5):
    kf = KFold(n_splits=n_splits)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = accuracy_score(y_test, predictions)
        scores.append(score)

    return np.mean(scores), scores

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def save_predictions_to_csv(original_values, predictions, filename='predictions.csv'):
    df = pd.DataFrame({'Original Values': original_values, 'Predictions': predictions})
    df.to_csv(filename, index=False)

def main():
    df = load_data()
    X, y = reshape_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model_rf = build_model('RandomForest')
    model_xgb = build_model('XGBoost')

    rf_score, rf_scores = k_fold_cross_validation(X, y, model_rf)
    xgb_score, xgb_scores = k_fold_cross_validation(X, y, model_xgb)

    print(f"RandomForest Average Score: {rf_score}")
    print(f"XGBoost Average Score: {xgb_score}")

    # Train on the whole dataset and make predictions for future use
    model_rf.fit(X, y)
    model_xgb.fit(X, y)

    rf_predictions = model_rf.predict(X)
    xgb_predictions = model_xgb.predict(X)

    save_predictions_to_csv(y, rf_predictions, 'rf_predictions.csv')
    save_predictions_to_csv(y, xgb_predictions, 'xgb_predictions.csv')

    # Save the RandomForest model to a pickle file
    with open('random_forest_model.pkl', 'wb') as file:
        pickle.dump(model_rf, file)

    print("RandomForest model saved to random_forest_model.pkl")

if __name__ == '__main__':
    main()