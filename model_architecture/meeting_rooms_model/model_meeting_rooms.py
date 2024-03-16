# # from datetime import datetime
# #
# # import pandas as pd
# # import numpy as np
# # import csv
# # import joblib
# # from keras import Model, Input
# #
# # from keras.models import Sequential
# # from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, LSTM, LeakyReLU, BatchNormalization
# # from keras.optimizers.legacy import Adam
# # from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, LambdaCallback
# # from keras.regularizers import l1_l2, l2
# # from sklearn.compose import ColumnTransformer
# # from sklearn.metrics import accuracy_score
# # from sklearn.model_selection import train_test_split, KFold, GridSearchCV
# # import matplotlib.pyplot as plt
# # from sklearn.multioutput import MultiOutputClassifier
# # from sklearn.preprocessing import StandardScaler, OneHotEncoder
# # import xgboost as xgb
# # from sklearn.svm import SVC
# # from xgboost import XGBClassifier
# #
# # np.set_printoptions(precision=2)
# #
# #
# # def plot_history(history):
# #     plt.plot(history.history['loss'], label='train')
# #     plt.plot(history.history['val_loss'], label='validation')
# #     plt.title('Model loss')
# #     plt.ylabel('Loss')
# #     plt.xlabel('Epoch')
# #     plt.legend(['Train', 'Validation'], loc='upper left')
# #     plt.show()
# #
# #
# # def load_data(filepath):
# #     df = pd.read_csv(filepath)
# #
# #     # Preprocessing steps
# #     features = df.drop(
# #         ['date', 'attendanceNineToEleven', 'attendanceElevenToOne', 'attendanceOneToThree', 'attendanceThreeToFive'],
# #         axis=1)
# #     targets = df[['nineToEleven', 'elevenToOne', 'oneToThree', 'threeToFive']]
# #
# #     # Define numerical and categorical columns
# #     num_cols = ['capacity', 'day_of_week', 'month', 'week_of_year']
# #     cat_cols = ['room']
# #
# #     preprocessor = ColumnTransformer(
# #         transformers=[
# #             ('num', StandardScaler(), num_cols),
# #             ('cat', OneHotEncoder(), cat_cols)
# #         ]
# #     )
# #
# #     X = preprocessor.fit_transform(features)
# #     y = targets.to_numpy()
# #
# #     return X, y
# #
# #
# # def split_data(X, y, test_size=0.1, random_state=0):
# #     """
# #     Splits the data into training and testing sets.
# #
# #     Parameters:
# #     - X: Input features.
# #     - y: Target labels.
# #     - test_size: Proportion of the dataset to include in the test split.
# #     - random_state: Controls the shuffling applied to the data before applying the split.
# #
# #     Returns:
# #     - Training and testing sets.
# #     """
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
# #
# #     return np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32), \
# #         np.array(X_test, dtype=np.float32), np.array(y_test, dtype=np.float32)
# #
# #
# # def predict_and_save(model, X_test, y_test, filename='predictions.csv'):
# #     predictions = model.predict(X_test)
# #
# #     # Save predictions alongside actual values
# #     predictions_df = pd.DataFrame(predictions,
# #                                   columns=['PredictedNineToEleven', 'PredictedElevenToOne', 'PredictedOneToThree',
# #                                            'PredictedThreeToFive'])
# #     actual_df = pd.DataFrame(y_test, columns=['ActualNineToEleven', 'ActualElevenToOne', 'ActualOneToThree',
# #                                               'ActualThreeToFive'])
# #
# #     combined_df = pd.concat([actual_df, predictions_df], axis=1)
# #     combined_df.to_csv(filename, index=False)
# #
# #
# # def evaluate_model(model, X_test, y_test):
# #     predictions = model.predict(X_test)
# #     for i, label in enumerate(['NineToEleven', 'ElevenToOne', 'OneToThree', 'ThreeToFive']):
# #         accuracy = accuracy_score(y_test[:, i], predictions[:, i])
# #         print(f'Accuracy for {label}: {accuracy:.4f}')
# #
# #
# # def k_fold_cross_validation(X, y, n_splits=5):
# #     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
# #     fold_no = 1
# #     losses = []
# #     accuracies = []
# #
# #     for train_index, val_index in kf.split(X):
# #         print(f'Training on fold {fold_no}...')
# #
# #         # Splitting data
# #         x_train_fold, x_val_fold = X[train_index], X[val_index]
# #         y_train_fold, y_val_fold = y[train_index], y[val_index]
# #
# #         # Build model
# #         model = build_model(x_train_fold.shape[1])
# #         model.summary()
# #
# #         # Train model
# #         history = train_model(model, x_train_fold, y_train_fold, x_val_fold, y_val_fold, batch_size=5)
# #
# #         # Saving history
# #         losses.append(history.history['val_loss'][-1])
# #         accuracies.append(history.history['val_accuracy'][-1])
# #
# #         fold_no += 1
# #
# #     return losses, accuracies
# #
# #
# # def build_model():
# #     # Initialize the base classifier
# #     # base_svc = SVC(kernel='rbf', C=1.0, gamma='scale', decision_function_shape='ovr', probability=True)
# #
# #     xgb_model = XGBClassifier(objective='binary:logistic', n_estimators=100, learning_rate=0.05, max_depth=4)
# #     multioutput_model = MultiOutputClassifier(xgb_model, n_jobs=-1)
# #
# #     # Wrap it with MultiOutputClassifier to handle multi-label tasks
# #
# #     return multioutput_model
# #
# #
# # # def train_model(model, x_train, y_train):
# # #     model.fit(x_train, y_train)
# # #     return model
# #
# # def train_model(x_train, y_train):
# #     xgb_model = XGBClassifier()
# #     parameters = {
# #         'estimator__n_estimators': [100, 125, 150, 175, 200],
# #         'estimator__learning_rate': [0.01, 0.05, 0.1],
# #         'estimator__max_depth': [4, 5, 6],
# #         'estimator__subsample': [0.7, 0.8, 0.9, 1],
# #         'estimator__colsample_bytree': [0.7, 0.8, 0.9, 1],
# #     }
# #
# #     multioutput_model = MultiOutputClassifier(xgb_model, n_jobs=-1)
# #     clf = GridSearchCV(multioutput_model, parameters, scoring='accuracy', cv=3)
# #     clf.fit(x_train, y_train)
# #     print(f"Best parameters: {clf.best_params_}")
# #     print(f"Best CV score: {clf.best_score_}")
# #     return clf.best_estimator_
# #
# #
# # def save_model_to_pickle():
# #     pass
# #
# #
# # def main():
# #     print("Loading data...")
# #     X, y = load_data('preprocessed_meeting_room_data.csv')
# #     x_train, y_train, x_test, y_test = split_data(X, y)
# #
# #     print("Building Model...")
# #     model = build_model()
# #     print(model)
# #
# #     print("Training model...")
# #     # model = train_model(model, x_train, y_train)
# #     model = train_model(x_train, y_train)
# #
# #     print("Predicting and saving results...")
# #     predict_and_save(model, x_test, y_test, 'model_predictions.csv')
# #
# #     print("Evaluating model...")
# #     evaluate_model(model, x_test, y_test)
# #
# #
# # if __name__ == '__main__':
# #     main()
#
# # from sklearn.model_selection import train_test_split, GridSearchCV
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.preprocessing import StandardScaler, OneHotEncoder
# # from sklearn.compose import ColumnTransformer
# # from sklearn.metrics import classification_report
# # import pandas as pd
# # import numpy as np
# # import joblib  # For model saving
# #
# # def load_and_preprocess_data(filepath):
# #     df = pd.read_csv(filepath)
# #
# #     features = df.drop(['date', 'attendanceNineToEleven', 'attendanceElevenToOne', 'attendanceOneToThree', 'attendanceThreeToFive'], axis=1)
# #     targets = df[['nineToEleven', 'elevenToOne', 'oneToThree', 'threeToFive']].values
# #
# #     num_cols = ['capacity', 'day_of_week', 'month', 'week_of_year']
# #     cat_cols = ['room']
# #
# #     preprocessor = ColumnTransformer(
# #         transformers=[
# #             ('num', StandardScaler(), num_cols),
# #             ('cat', OneHotEncoder(), cat_cols)
# #         ])
# #
# #     X = preprocessor.fit_transform(features)
# #     y = targets
# #     return X, y
# #
# # def perform_grid_search(X_train, y_train):
# #     param_grid = {
# #         'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500],
# #         'max_depth': [None, 10, 15, 20, 25, 30, 35, 45, 50],
# #         'criterion': ['gini', 'entropy']
# #     }
# #     grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='f1_weighted')
# #     grid_search.fit(X_train, y_train)
# #     return grid_search.best_estimator_
# #
# # def evaluate_and_save_predictions(model, X_test, y_test, filename='predictions.csv'):
# #     predictions = model.predict(X_test)
# #     # Modify this part to handle multi-output format properly
# #     print("Classification Report for Each Label:")
# #     for i, column in enumerate(['nineToEleven', 'elevenToOne', 'oneToThree', 'threeToFive']):
# #         print(f"Report for {column}:")
# #         print(classification_report(y_test[:, i], predictions[:, i]))
# #
# #     # Save predictions
# #     np.savetxt(filename, np.hstack((y_test, predictions)), delimiter=",")
# #
# # def main():
# #     filepath = 'preprocessed_meeting_room_data.csv'
# #     X, y = load_and_preprocess_data(filepath)
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #
# #     # Model training with Grid Search
# #     best_model = perform_grid_search(X_train, y_train)
# #
# #     # Evaluating and saving predictions
# #     evaluate_and_save_predictions(best_model, X_test, y_test, 'model_predictions.csv')
# #
# #     # Optionally, save the trained model
# #     joblib.dump(best_model, 'best_random_forest_model.pkl')
# #
# # if __name__ == '__main__':
# #     main()
#
#
# import pandas as pd
# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.src.callbacks import ReduceLROnPlateau
# from keras.src.optimizers import Adam
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib.pyplot as plt
# from xgboost.callback import EarlyStopping
#
#
# def load_data(filepath):
#     df = pd.read_csv(filepath)
#     # Aggregating attendance across different times
#     df['total_attendance'] = df[
#         ['attendanceNineToEleven', 'attendanceElevenToOne', 'attendanceOneToThree', 'attendanceThreeToFive']].sum(
#         axis=1)
#     X = df.drop(['row', 'total_attendance', 'date'], axis=1)
#     y = df['total_attendance']
#     return X, y
#
#
# def preprocess_data(X):
#     numerical_features = ['capacity']
#     categorical_features = X.drop(columns=numerical_features).columns.tolist()
#
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', StandardScaler(), numerical_features),
#             ('cat', OneHotEncoder(), categorical_features)
#         ])
#
#     X_processed = preprocessor.fit_transform(X)
#     # Convert sparse matrix to dense
#     X_processed = X_processed.toarray()
#     return X_processed, preprocessor
#
#
# def build_model(input_shape):
#     model = Sequential([
#         Dense(64, input_dim=input_shape, activation='relu'),
#         Dense(32, activation='relu'),
#         Dense(1, activation='linear')  # Output layer for regression
#     ])
#
#     optimizer = Adam(learning_rate=0.005)
#
#     model.compile(optimizer=optimizer, loss='mse', metrics='mae')
#     return model
#
#
# def plot_history(history):
#     plt.plot(history.history['loss'], label='loss')
#     plt.plot(history.history['val_loss'], label='val_loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Error [Total Attendance]')
#     plt.legend()
#     plt.show()
#
#
# def train_model(model, x_train, y_train, x_val, y_val, batch_size):
#     # early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
#     learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
#
#     history = model.fit(
#         x_train, y_train, epochs=1000,
#         verbose=1,
#         validation_data=(x_val, y_val),
#         callbacks=[learning_rate_reduction],
#         batch_size=batch_size
#     )
#
#     avg_mae = np.mean(history.history['mae'])
#     print(f"Average MAE: {avg_mae}")
#
#     plot_history(history)
#
#
# def main():
#     filepath = 'preprocessed_meeting_room_data.csv'  # Update this path
#     X, y = load_data(filepath)
#     X_processed, preprocessor = preprocess_data(X)
#
#     X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
#
#     model = build_model(X_train.shape[1])
#     train_model(model, X_train, y_train, X_test, y_test, batch_size=5)
#
#     # y_pred = model.predict(X_test)
#     # mse = mean_squared_error(y_test, y_pred)
#     # r2 = r2_score(y_test, y_pred)
#     # print(f'MSE: {mse}, R2: {r2}')
#
#
# if __name__ == '__main__':
#     main()

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score, KFold
# from sklearn.metrics import mean_squared_error
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
# import pickle
#
# np.set_printoptions(precision=2)
#
# # Load data
# def load_data(filepath):
#     df = pd.read_csv(filepath)
#     # Assuming the sum of 'attendance*' columns gives the total attendance for each room
#     df['total_attendance'] = df.filter(like='attendance').sum(axis=1)
#     # Aggregate by date to get total attendance per day
#     daily_attendance = df.groupby('date')['total_attendance'].sum().reset_index()
#     # You might want to add more features based on the date, such as day of the week, month, etc., if not already included
#     # Convert date to datetime to extract features
#     daily_attendance['date'] = pd.to_datetime(daily_attendance['date'])
#     daily_attendance['day_of_week'] = daily_attendance['date'].dt.dayofweek
#     daily_attendance['month'] = daily_attendance['date'].dt.month
#     # Add any other features you think might be relevant
#     return daily_attendance
#
# # Split data into features and target
# def split_data(data):
#     X = data[['day_of_week', 'month']]  # Add more features as needed
#     y = data['total_attendance']
#     return train_test_split(X, y, test_size=0.1, random_state=42)
#
# # Build the model
# def build_model():
#     return LinearRegression()
#
# # Train the model
# def train_model(model, X_train, y_train):
#     model.fit(X_train, y_train)
#     return model
#
# # Evaluate the model
# # Evaluate the model and save predictions
# def evaluate_model_and_save_predictions(model, X_test, y_test, filename='predictions.csv'):
#     predictions = model.predict(X_test)
#     mse = mean_squared_error(y_test, predictions)
#     print(f'Mean Squared Error: {mse}')
#     # Save predictions and actual values to CSV
#     save_predictions_to_csv(y_test, predictions, filename)
#     return mse
#
# # Save real and predicted values to a CSV file
# def save_predictions_to_csv(y_test, predictions, filename='predictions.csv'):
#     predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
#     predictions_df.to_csv(filename, index=False)
#     print(f'Saved predictions to {filename}')
# # Save the model to a pickle file
# def save_model_to_pickle(model, filename='model.pkl'):
#     with open(filename, 'wb') as file:
#         pickle.dump(model, file)
#
# def main():
#     data = load_data('preprocessed_meeting_room_data.csv')
#     X_train, X_test, y_train, y_test = split_data(data)
#     model = build_model()
#     trained_model = train_model(model, X_train, y_train)
#     evaluate_model_and_save_predictions(trained_model, X_test, y_test)
#     save_model_to_pickle(trained_model)
#
# if __name__ == '__main__':
#     main()

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import matplotlib.pyplot as plt
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

    # Preparing input features.
    X = df[['capacity', 'day_of_week', 'month']]

    # Preparing separate target variables for each time slot.
    y = df[['nineToEleven', 'elevenToOne']]

    return X, y


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

def main():
    filepath = 'preprocessed_month_categorical.csv'
    X, y = load_data(filepath)

    models = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    y_test_results = pd.DataFrame(index=X_test.index)

    for time_slot in y.columns:
        model = SVC()
        model.fit(X_train, y_train[time_slot])
        models[time_slot] = model

        print(f"Evaluating model for {time_slot}:")
        evaluate_model(model, X_test, y_test[time_slot], time_slot)

        y_test_results[f'Predicted_{time_slot}'] = model.predict(X_test)

    predictions_filename = 'predictions_test.csv'
    y_test_results.to_csv(predictions_filename, index=False)
    print(f"Predictions saved to {predictions_filename}")


# def main():
#     filepath = 'preprocessed_month_categorical.csv'
#     X, y = load_data(filepath)
#
#     models = {}
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
#     y_test_results = pd.DataFrame(index=X_test.index)
#
#     for time_slot in y.columns:
#         # Perform grid search to find the best model
#         best_model = perform_grid_search(X_train, y_train[time_slot])
#         models[time_slot] = best_model
#
#         print(f"Evaluating model for {time_slot}:")
#         evaluate_model(best_model, X_test, y_test[time_slot], time_slot)
#
#         y_test_results[f'Predicted_{time_slot}'] = best_model.predict(X_test)
#
#     predictions_filename = 'predictions_test.csv'
#     y_test_results.to_csv(predictions_filename, index=False)
#     print(f"Predictions saved to {predictions_filename}")
#
# if __name__ == '__main__':
#     main()

if __name__ == '__main__':
    main()
