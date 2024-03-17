import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


def load_data(filepath):
    data = pd.read_csv(filepath)
    # Convert date to datetime and extract features if needed
    data['date'] = pd.to_datetime(data['date'])
    # Assuming 'availability' is the target variable, which needs to be defined based on your criteria (e.g., if a meeting room is booked in any of the slots, it's not available)
    return data


def preprocess_data(data):
    # Convert categorical data to numerical if needed, e.g., room names to categorical codes
    data['room'] = data['room'].astype('category').cat.codes
    # Here, define your 'y' based on the meeting room availability; this is an example
    # You might need to adjust based on your actual availability criteria
    data['availability'] = 1 - data[['nineToEleven', 'elevenToOne', 'oneToThree', 'threeToFive']].max(axis=1)
    X = data[['day_of_week', 'month', 'week_of_year', 'room']]
    y = data['availability']
    return X, y


def build_and_train_model(X_train, y_train):
    # Example model: RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")


def main():
    data = load_data('merged_data.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_and_train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Feature Importance
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    feature_importances.nlargest(10).plot(kind='barh')
    plt.show()

    # Save the model to a file
    with open('model_test_fewer_features.pkl', 'wb') as file:
        pickle.dump(model, file)
    print("Model saved to employee_count_prediction_model.pkl")


if __name__ == '__main__':
    main()
# import pickle
#
# # import numpy as np
# # import pandas as pd
# # from sklearn.model_selection import train_test_split, GridSearchCV
# # from sklearn.metrics import accuracy_score
# # from sklearn.ensemble import RandomForestClassifier
# # from xgboost import XGBClassifier
# # import matplotlib.pyplot as plt
# #
# # def load_data(filepath):
# #     data = pd.read_csv(filepath)
# #     # Convert date to datetime and extract features if needed
# #     data['date'] = pd.to_datetime(data['date'])
# #     return data
# #
# # def preprocess_data(data):
# #     # Example of adding a new feature: number of booked intervals per day
# #     data['num_booked_intervals'] = data[['nineToEleven', 'elevenToOne', 'oneToThree', 'threeToFive']].sum(axis=1)
# #     data['room'] = data['room'].astype('category').cat.codes
# #     X = data[['capacity', 'day_of_week', 'month', 'week_of_year',
# #               'total_booked_desks_first_half', 'total_booked_desks_second_half',
# #               'room', 'num_booked_intervals']]
# #     y = data[['nineToEleven', 'elevenToOne', 'oneToThree', 'threeToFive']]
# #     return X, y
# # def build_and_train_model(X_train, y_train):
# #     # Tuning RandomForest parameters example
# #     model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
# #     model.fit(X_train, y_train)
# #     return model
# #
# # def evaluate_model(model, X_test, y_test):
# #     predictions = model.predict(X_test)
# #     # Assuming binary 'availability', we calculate accuracy for each interval
# #     accuracies = [accuracy_score(y_test.iloc[:, i], predictions[:, i]) for i in range(y_test.shape[1])]
# #     for i, acc in enumerate(accuracies):
# #         print(f"Interval {y_test.columns[i]} Accuracy: {acc}")
# #
# # def main():
# #     data = load_data('merged_data.csv')
# #     X, y = preprocess_data(data)
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #
# #     model = build_and_train_model(X_train, y_train)
# #     evaluate_model(model, X_test, y_test)
# #
# #     # For RandomForest, showing feature importances
# #     feature_importances = pd.Series(model.feature_importances_, index=X.columns)
# #     feature_importances.nlargest(10).plot(kind='barh')
# #     plt.show()
# #
# # if __name__ == '__main__':
# #     main()
#
#
# # import pandas as pd
# # import numpy as np
# # from sklearn.model_selection import train_test_split, KFold
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.svm import SVC
# # from xgboost import XGBClassifier
# # from sklearn.metrics import accuracy_score, confusion_matrix
# # import matplotlib.pyplot as plt
# #
# #
# # def plot_history(histories, key='accuracy'):
# #     plt.figure(figsize=(16, 10))
# #     for name, history in histories:
# #         val = plt.plot(history.epoch, history.history['val_' + key],
# #                        '--', label=name.title() + ' Val')
# #         plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
# #                  label=name.title() + ' Train')
# #     plt.xlabel('Epochs')
# #     plt.ylabel(key.replace('_', ' ').title())
# #     plt.legend()
# #     plt.xlim([0, max(history.epoch)])
#
#
# # def load_data():
# #     df = pd.read_csv('merged_data.csv')
# #     # Assuming 'available' is a binary target feature indicating room availability
# #     df['available'] = df.apply(
# #         lambda row: 1 if row['nineToEleven'] + row['elevenToOne'] + row['oneToThree'] + row['threeToFive'] > 0 else 0,
# #         axis=1)
# #     return df
#
# # def load_data(filepath):
# #     df = pd.read_csv(filepath)
# #
# #     # Preparing input features.
# #     X = df[['capacity', 'day_of_week', 'month', 'week_of_year', 'total_booked_desks_first_half',
# #               'total_booked_desks_second_half', 'room']]
# #
# #     # Preparing separate target variables for each time slot.
# #     y = df[['nineToEleven', 'elevenToOne', 'oneToThree', 'threeToFive']]
# #
# #     return X, y
#
# # def load_data(filepath):
# #     df = pd.read_csv(filepath)
# #
# #     # One-hot encode the 'room' feature
# #     df = pd.get_dummies(df, columns=['room'])
# #
# #     # Selecting relevant features
# #     X = df[['capacity', 'month', 'week_of_year', 'total_booked_desks_first_half',
# #             'total_booked_desks_second_half'] + [col for col in df.columns if col.startswith('room_')]]
# #
# #     # Preparing separate target variables for each time slot
# #     y = df[['nineToEleven', 'elevenToOne', 'oneToThree', 'threeToFive']]
# #
# #     return X, y
# #
# #
# #
# # def save_predictions_to_csv(original_values, predictions, filename='predictions.csv'):
# #     df = pd.DataFrame({'Original Values': original_values, 'Predictions': predictions})
# #     df.to_csv(filename, index=False)
# #
# #
# #
# # def build_model(model_type='RandomForest'):
# #     if model_type == 'RandomForest':
# #         return RandomForestClassifier()
# #     elif model_type == 'XGBoost':
# #         return XGBClassifier(use_label_encoder=False, eval_metric='logloss')
# #
# #
# # def k_fold_cross_validation(X, y, model, n_splits=5):
# #     kf = KFold(n_splits=n_splits)
# #     scores = []
# #
# #     for train_index, test_index in kf.split(X):
# #         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
# #         y_train, y_test = y.iloc[train_index], y.iloc[test_index]
# #
# #         model.fit(X_train, y_train)
# #         predictions = model.predict(X_test)
# #         score = accuracy_score(y_test, predictions)
# #         scores.append(score)
# #
# #     return np.mean(scores), scores
# #
# #
# # def split_data(X, y):
# #     return train_test_split(X, y, test_size=0.2, random_state=42)
# #
# #
# #
# #
# # # def main():
# # #     filepath = 'merged_data.csv'
# # #     X, y = load_data(filepath)
# # #     # X, y = reshape_data(df)
# # #     X_train, X_test, y_train, y_test = split_data(X, y)
# # #
# # #     model_rf = build_model('RandomForest')
# # #     model_xgb = build_model('XGBoost')
# # #
# # #     rf_score, rf_scores = k_fold_cross_validation(X, y, model_rf)
# # #     xgb_score, xgb_scores = k_fold_cross_validation(X, y, model_xgb)
# # #
# # #     print(f"RandomForest Average Score: {rf_score}")
# # #     print(f"XGBoost Average Score: {xgb_score}")
# # #
# # #     # Train on the whole dataset and make predictions for future use
# # #     model_rf.fit(X, y)
# # #     model_xgb.fit(X, y)
# # #
# # #     rf_predictions = model_rf.predict(X)
# # #     xgb_predictions = model_xgb.predict(X)
# # #
# # #     save_predictions_to_csv(y, rf_predictions, 'rf_predictions.csv')
# # #     save_predictions_to_csv(y, xgb_predictions, 'xgb_predictions.csv')
# #
# # def evaluate_model(model, X_test, y_test, time_slot):
# #     predictions = model.predict(X_test)
# #     accuracy = accuracy_score(y_test, predictions)
# #     print(f'{time_slot} Accuracy: {accuracy}')
# #     print(confusion_matrix(y_test, predictions))
# #
# # def main():
# #     filepath = 'merged_data.csv'
# #     X, y = load_data(filepath)
# #
# #     models = {}
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
# #     y_test_results = pd.DataFrame(index=X_test.index)
# #
# #     for time_slot in y.columns:
# #         model = SVC()
# #         model.fit(X_train, y_train[time_slot])
# #         models[time_slot] = model
# #
# #         print(f"Evaluating model for {time_slot}:")
# #         evaluate_model(model, X_test, y_test[time_slot], time_slot)
# #
# #         y_test_results[f'Predicted_{time_slot}'] = model.predict(X_test)
# #
# #     predictions_filename = 'predictions_test2.csv'
# #     y_test_results.to_csv(predictions_filename, index=False)
# #     print(f"Predictions saved to {predictions_filename}")
# #
# #
# # if __name__ == '__main__':
# #     main()
#
# # import numpy as np
# # import pandas as pd
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GridSearchCV
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.metrics import accuracy_score, confusion_matrix
# # import pickle
# # import matplotlib.pyplot as plt
# # from xgboost import XGBClassifier
# # from sklearn.svm import SVC
# #
# #
# #
# # # Function to plot the history of model accuracy
# # def plot_history(history):
# #     plt.figure(figsize=(10, 5))
# #     plt.plot(history['train_acc'], label='Train Accuracy')
# #     plt.plot(history['val_acc'], label='Validation Accuracy')
# #     plt.xlabel('Epoch')
# #     plt.ylabel('Accuracy')
# #     plt.title('Model Accuracy Over Epochs')
# #     plt.legend()
# #     plt.show()
# #
# #
# #
# #
# # def load_data(filepath):
# #     df = pd.read_csv(filepath)
# #
# #     # Preparing input features.
# #     X = df[['capacity', 'day_of_week', 'month', 'week_of_year', 'total_booked_desks_first_half',
# #               'total_booked_desks_second_half', 'room']]
# #
# #     # Preparing separate target variables for each time slot.
# #     y = df[['nineToEleven', 'elevenToOne', 'oneToThree', 'threeToFive']]
# #
# #     return X, y
# #
# #
# # # Function to split data into training and testing sets
# # def split_data(X, y, test_size=0.1, random_state=0):
# #     return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
# #
# #
# #
# #
# # def predict_and_save(models, X_test, y_test, filename='predictions_test.csv'):
# #     results = pd.DataFrame(y_test)
# #     for time_slot in models:
# #         model = models[time_slot]
# #         results[f'Predicted_{time_slot}'] = model.predict(X_test)
# #     results.to_csv(filename, index=False)
# #     print("Predictions saved to", filename)
# #
# # def build_xgboost_model():
# #     return XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0)
# #
# # def evaluate_model(model, X_test, y_test, time_slot):
# #     predictions = model.predict(X_test)
# #     accuracy = accuracy_score(y_test, predictions)
# #     print(f'{time_slot} Accuracy: {accuracy}')
# #     print(confusion_matrix(y_test, predictions))
# #
# #
# # # Function to perform k-fold cross-validation
# # def k_fold_cross_validation(X, y, n_splits=5):
# #     skf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
# #     fold_no = 1
# #     for train_index, test_index in skf.split(X, y):
# #         X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
# #         y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
# #
# #         model = build_xgboost_model()
# #         model.fit(X_train_fold, y_train_fold)
# #         predictions = model.predict(X_test_fold)
# #
# #         accuracy = accuracy_score(y_test_fold, predictions)
# #         print(f'Fold {fold_no}, Accuracy: {accuracy}')
# #         fold_no += 1
# #
# #
# # # Function to build and return the model
# # def build_model():
# #     return RandomForestClassifier(random_state=0)
# #
# # def perform_grid_search(X_train, y_train):
# #     # Define a parameter grid to search
# #     param_grid = {
# #         'n_estimators': [100, 175, 200],  # Number of trees in the forest
# #         'learning_rate': [0.01, 0.1],  # Step size shrinkage used to prevent overfitting
# #         'max_depth': [3, 4, 5, 7],  # Maximum depth of a tree
# #         'subsample': [0.8, 0.9, 1],  # Subsample ratio of the training instances
# #         'colsample_bytree': [0.8, 1],  # Subsample ratio of columns when constructing each tree
# #     }
# #
# #     # Initialize the XGBClassifier
# #     xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0)
# #
# #     # Setup GridSearchCV
# #     grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=8, n_jobs=-1, verbose=1)
# #
# #     # Perform the grid search on the training data
# #     grid_search.fit(X_train, y_train)
# #
# #     # Return the best estimator
# #     return grid_search.best_estimator_
# #
# # def main():
# #     filepath = 'merged_data.csv'
# #     X, y = load_data(filepath)
# #
# #     models = {}
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
# #     y_test_results = pd.DataFrame(index=X_test.index)
# #
# #     for time_slot in y.columns:
# #         model = SVC()
# #         model.fit(X_train, y_train[time_slot])
# #         models[time_slot] = model
# #
# #         print(f"Evaluating model for {time_slot}:")
# #         evaluate_model(model, X_test, y_test[time_slot], time_slot)
# #
# #         y_test_results[f'Predicted_{time_slot}'] = model.predict(X_test)
# #
# #     predictions_filename = 'predictions_test.csv'
# #     y_test_results.to_csv(predictions_filename, index=False)
# #     print(f"Predictions saved to {predictions_filename}")
# #
# #
# # # def main():
# # #     filepath = 'preprocessed_month_categorical.csv'
# # #     X, y = load_data(filepath)
# # #
# # #     models = {}
# # #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
# # #     y_test_results = pd.DataFrame(index=X_test.index)
# # #
# # #     for time_slot in y.columns:
# # #         # Perform grid search to find the best model
# # #         best_model = perform_grid_search(X_train, y_train[time_slot])
# # #         models[time_slot] = best_model
# # #
# # #         print(f"Evaluating model for {time_slot}:")
# # #         evaluate_model(best_model, X_test, y_test[time_slot], time_slot)
# # #
# # #         y_test_results[f'Predicted_{time_slot}'] = best_model.predict(X_test)
# # #
# # #     predictions_filename = 'predictions_test.csv'
# # #     y_test_results.to_csv(predictions_filename, index=False)
# # #     print(f"Predictions saved to {predictions_filename}")
# # #
# # # if __name__ == '__main__':
# # #     main()
# #
# # if __name__ == '__main__':
# #     main()
#
# #
# # import pandas as pd
# # import numpy as np
# # from sklearn.model_selection import train_test_split, KFold
# # from sklearn.ensemble import RandomForestClassifier
# # from xgboost import XGBClassifier
# # from sklearn.metrics import accuracy_score
# # import matplotlib.pyplot as plt
# #
# # def plot_history(histories, key='accuracy'):
# #     plt.figure(figsize=(16, 10))
# #     for name, history in histories:
# #         val = plt.plot(history.epoch, history.history['val_' + key],
# #                        '--', label=name.title() + ' Val')
# #         plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
# #                  label=name.title() + ' Train')
# #     plt.xlabel('Epochs')
# #     plt.ylabel(key.replace('_', ' ').title())
# #     plt.legend()
# #     plt.xlim([0, max(history.epoch)])
# #
# # # def load_data():
# # #     df = pd.read_csv('merged_data.csv')
# # #     # Assuming 'available' is a binary target feature indicating room availability
# # #     df['available'] = df.apply(
# # #         lambda row: 1 if row['nineToEleven'] + row['elevenToOne'] + row['oneToThree'] + row['threeToFive'] > 0 else 0,
# # #         axis=1)
# # #     return df
# #
# #
# # def load_data():
# #     df = pd.read_csv('merged_data.csv')
# #     # Convert each time slot to a binary target variable
# #     df['nineToEleven_available'] = df['nineToEleven'].apply(lambda x: 1 if x > 0 else 0)
# #     df['elevenToOne_available'] = df['elevenToOne'].apply(lambda x: 1 if x > 0 else 0)
# #     df['oneToThree_available'] = df['oneToThree'].apply(lambda x: 1 if x > 0 else 0)
# #     df['threeToFive_available'] = df['threeToFive'].apply(lambda x: 1 if x > 0 else 0)
# #     return df
# #
# #
# # # def reshape_data(df):
# # #     # Select relevant features
# # #     X = df[['capacity', 'day_of_week', 'month', 'week_of_year', 'total_booked_desks_first_half',
# # #             'total_booked_desks_second_half']]
# # #     y = df['available']
# # #     return X, y
# #
# # def reshape_data(df):
# #     X = df[['capacity', 'day_of_week', 'month', 'week_of_year', 'total_booked_desks_first_half', 'total_booked_desks_second_half']]
# #     y = df[['nineToEleven_available', 'elevenToOne_available', 'oneToThree_available', 'threeToFive_available']]
# #     return X, y
# #
# # def build_model(model_type='RandomForest'):
# #     if model_type == 'RandomForest':
# #         return RandomForestClassifier(n_jobs=-1, random_state=42)
# #     elif model_type == 'XGBoost':
# #         return XGBClassifier(use_label_encoder=False, eval_metric='logloss')
# #
# # def k_fold_cross_validation(X, y, model, n_splits=5):
# #     kf = KFold(n_splits=n_splits)
# #     scores = []
# #
# #     for train_index, test_index in kf.split(X):
# #         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
# #         y_train, y_test = y.iloc[train_index], y.iloc[test_index]
# #
# #         model.fit(X_train, y_train)
# #         predictions = model.predict(X_test)
# #         score = accuracy_score(y_test, predictions)
# #         scores.append(score)
# #
# #     return np.mean(scores), scores
# #
# # def split_data(X, y):
# #     return train_test_split(X, y, test_size=0.2, random_state=42)
# #
# # def save_predictions_to_csv(original_values, predictions, filename='predictions.csv'):
# #     df = pd.DataFrame({'Original Values': original_values, 'Predictions': predictions})
# #     df.to_csv(filename, index=False)
# #
# # def main():
# #     df = load_data()
# #     X, y = reshape_data(df)
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
# #
# #     # Save the RandomForest model to a pickle file
# #     # with open('random_forest_model.pkl', 'wb') as file:
# #     #     pickle.dump(model_rf, file)
# #
# #     print("RandomForest model saved to random_forest_model.pkl")
# #
# # if __name__ == '__main__':
# #     main()
#
#
# # import pandas as pd
# # import numpy as np
# # import tensorflow as tf
# # from keras.src.callbacks import ReduceLROnPlateau
# # from keras.src.optimizers import Adam
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import StandardScaler
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import Dense, Dropout
# # from tensorflow.keras.callbacks import EarlyStopping
# # import matplotlib.pyplot as plt
# #
# #
# # def plot_history(history):
# #     plt.figure(figsize=(16, 10))
# #     plt.plot(history.history['accuracy'], label='accuracy')
# #     plt.plot(history.history['val_accuracy'], label='val_accuracy')
# #     plt.xlabel('Epoch')
# #     plt.ylabel('Accuracy')
# #     plt.ylim([0, 1])
# #     plt.legend(loc='lower right')
# #
# #
# # def load_data():
# #     df = pd.read_csv('merged_data.csv')
# #     return df
# #
# #
# # def reshape_data(df):
# #     X = df[['capacity', 'day_of_week', 'month', 'week_of_year', 'total_booked_desks_first_half',
# #             'total_booked_desks_second_half']].values
# #     y = df[['nineToEleven', 'elevenToOne', 'oneToThree', 'threeToFive']].values
# #     scaler = StandardScaler()
# #     X_scaled = scaler.fit_transform(X)
# #     return X_scaled, y
# #
# #
# # def build_model(input_shape):
# #     model = Sequential([
# #         Dense(64, activation='relu', input_shape=(input_shape,)),
# #         # Dropout(0.2),
# #         # Dense(128, activation='relu'),
# #         # Dropout(0.2),
# #         Dense(4, activation='sigmoid')  # 4 output units for each time interval
# #     ])
# #
# #     optimiser = Adam(learning_rate=0.005)
# #
# #     model.compile(optimizer='adam',
# #                   loss='binary_crossentropy',
# #                   metrics=['accuracy'])
# #     return model
# #
# #
# # def main():
# #     df = load_data()
# #     X, y = reshape_data(df)
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# #
# #     model = build_model(X_train.shape[1])
# #     # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# #     learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
# #
# #     history = model.fit(X_train, y_train, epochs=100, validation_split=0.2,callbacks=[learning_rate_reduction], verbose=1, batch_size=5)
# #
# #     test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
# #     print(f"\nTest accuracy: {test_acc}")
# #
# #     plot_history(history)
# #
# #     # Optional: Save the model for later use
# #     # model.save('meeting_room_booking_model.h5')
# #
# #
# # if __name__ == '__main__':
# #     main()
#
#
# # import pandas as pd
# # import numpy as np
# # from sklearn.model_selection import train_test_split, KFold, cross_val_score
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.multioutput import MultiOutputClassifier
# # from sklearn.metrics import accuracy_score
# # import matplotlib.pyplot as plt
# # import pickle
# #
# #
# # def load_data(filepath='merged_data.csv'):
# #     df = pd.read_csv(filepath)
# #     return df
# #
# #
# # def reshape_data(df):
# #     # Features for the model
# #     X = df[['capacity', 'day_of_week', 'month', 'week_of_year', 'total_booked_desks_first_half',
# #             'total_booked_desks_second_half']]
# #
# #     # Targets for the model: each time slot is a separate target
# #     y = df[['nineToEleven', 'elevenToOne', 'oneToThree', 'threeToFive']]
# #
# #     return X, y
# #
# #
# # def build_and_train_model(X, y):
# #     # MultiOutputClassifier can be used to wrap any classifier to predict multiple targets.
# #     model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
# #
# #     # Splitting the dataset into training and testing sets
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #
# #     # Training the model
# #     model.fit(X_train, y_train)
# #
# #     # Making predictions
# #     y_pred = model.predict(X_test)
# #
# #     # Evaluating the model
# #     accuracy = accuracy_score(y_test, y_pred)
# #     print(f"Model Accuracy: {accuracy}")
# #
# #     return model, X_train, X_test, y_train, y_test, y_pred
# #
# #
# # def save_model(model, filename='meeting_room_availability_model.pkl'):
# #     with open(filename, 'wb') as file:
# #         pickle.dump(model, file)
# #     print(f"Model saved to {filename}")
# #
# #
# # def main():
# #     # Load and prepare data
# #     df = load_data('merged_data.csv')
# #     X, y = reshape_data(df)
# #
# #     # Build and train the model
# #     model, X_train, X_test, y_train, y_test, y_pred = build_and_train_model(X, y)
# #
# #     # Save the trained model
# #     # save_model(model)
# #
# #
# # if __name__ == '__main__':
# #     main()
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix
# import pickle
# import matplotlib.pyplot as plt
# from sklearn.multioutput import MultiOutputClassifier
# from xgboost import XGBClassifier
# from sklearn.svm import SVC
#
#
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
#     # One-hot encoding the 'room' column
#     df = pd.get_dummies(df, columns=['room'])
#
#     # Preparing input features and target variables
#     X = df.drop(['row', 'date', 'nineToEleven', 'attendanceNineToEleven', 'elevenToOne', 'attendanceElevenToOne', 'oneToThree',
#                  'attendanceOneToThree', 'threeToFive', 'attendanceThreeToFive', 'week_of_year'], axis=1)
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
# def evaluate_multioutput_model(model, X_test, y_test):
#     predictions = model.predict(X_test)
#     accuracies = []
#     for i in range(y_test.shape[1]):  # Assuming y_test is a DataFrame
#         accuracy = accuracy_score(y_test.iloc[:, i], predictions[:, i])
#         accuracies.append(accuracy)
#         print(f'Accuracy for output {i+1}: {accuracy}')
#     return accuracies
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
#
#
#     # Return the best estimator
#     return grid_search.best_estimator_
#
#
# def predict_availability(models, date_features, room, time_slot):
#     # Prepare the input vector based on the given date and room
#     # Example date_features: {'day_of_week': 1, 'month': 5, 'total_booked_desks_first_half': 30, ...}
#     input_vector = date_features.copy()
#     input_vector.update(room)  # Assuming room is a dict like {'room_Pit-Lane': 1}
#
#     # Convert input_vector to DataFrame
#     input_df = pd.DataFrame([input_vector])
#
#     # Predict using the model for the specified time slot
#     prediction = models[time_slot].predict(input_df)
#
#     # Translate prediction to availability
#     availability = "Available" if prediction[0] == 0 else "Not Available"
#
#     return availability
#
#
#
# def build_and_train_multioutput_model(X_train, y_train):
#     # Base estimator
#     base_estimator = RandomForestClassifier(random_state=0)
#
#     # MultiOutputClassifier wrapper
#     multioutput_model = MultiOutputClassifier(base_estimator, n_jobs=-1)
#
#     # Training
#     multioutput_model.fit(X_train, y_train)
#
#     return multioutput_model
#
#
# def save_model_to_file(model, filename):
#     with open(filename, 'wb') as file:
#         pickle.dump(model, file)
#     print(f'Model saved to {filename}')
#
#
# def main():
#     filepath = 'merged_data.csv'
#     X, y = load_data(filepath)
#
#     # Splitting data
#     X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=0)
#
#     # Building and training the model
#     model = build_and_train_multioutput_model(X_train, y_train)
#
#     evaluate_multioutput_model(model, X_test, y_test)
#
#     model_filename = 'model_4outputs.pkl'
#     save_model_to_file(model, model_filename)
#
#
#
#
#
# if __name__ == '__main__':
#     main()



# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.model_selection import train_test_split
# import numpy as np
# import pandas as pd
# import pickle
#
# def load_data(filepath):
#     df = pd.read_csv(filepath)
#     df = pd.get_dummies(df, columns=['room'])
#     X = df.drop(['row', 'date', 'nineToEleven', 'attendanceNineToEleven', 'elevenToOne', 'attendanceElevenToOne', 'oneToThree',
#                  'attendanceOneToThree', 'threeToFive', 'attendanceThreeToFive', 'week_of_year'], axis=1)
#     # Use attendance as target variable
#     y = df[['attendanceNineToEleven', 'attendanceElevenToOne', 'attendanceOneToThree', 'attendanceThreeToFive']]
#     return X, y
#
# def build_and_train_model(X_train, y_train):
#     # Using RandomForestRegressor as an example
#     model = RandomForestRegressor(n_estimators=100, random_state=0)
#     model.fit(X_train, y_train)
#     return model
#
# def evaluate_regression_model(model, X_test, y_test):
#     predictions = model.predict(X_test)
#     mse = mean_squared_error(y_test, predictions)
#     mae = mean_absolute_error(y_test, predictions)
#     r2 = r2_score(y_test, predictions)
#     print(f"Mean Squared Error: {mse}")
#     print(f"Mean Absolute Error: {mae}")
#     print(f"R-squared: {r2}")
#
# def main():
#     filepath = 'merged_data.csv'
#     X, y = load_data(filepath)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = build_and_train_model(X_train, y_train)
#     evaluate_regression_model(model, X_test, y_test)
#
#     # Save the model to a file
#     # with open('employee_count_prediction_model.pkl', 'wb') as file:
#     #     pickle.dump(model, file)
#     # print("Model saved to employee_count_prediction_model.pkl")
#
# if __name__ == '__main__':
#     main()
