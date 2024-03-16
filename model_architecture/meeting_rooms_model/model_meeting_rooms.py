from datetime import datetime

import pandas as pd
import numpy as np
import csv
import joblib

from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, LSTM, LeakyReLU, BatchNormalization
from keras.optimizers.legacy import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, LambdaCallback
from keras.regularizers import l1_l2, l2
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb

# # # np.set_printoptions(precision=2)
# # #
# # #
def plot_history(history):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
# # #
# # #
# # # def load_data(filepath):
# # #     df = pd.read_csv(filepath)
# # #
# # #     # Preprocessing steps
# # #     features = df.drop(
# # #         ['date', 'attendanceNineToEleven', 'attendanceElevenToOne', 'attendanceOneToThree', 'attendanceThreeToFive'],
# # #         axis=1)
# # #     targets = df[['nineToEleven', 'elevenToOne', 'oneToThree', 'threeToFive']]
# # #
# # #     # Define numerical and categorical columns
# # #     num_cols = ['capacity', 'day_of_week', 'month', 'week_of_year']
# # #     cat_cols = ['room']
# # #
# # #     preprocessor = ColumnTransformer(
# # #         transformers=[
# # #             ('num', StandardScaler(), num_cols),
# # #             ('cat', OneHotEncoder(), cat_cols)
# # #         ]
# # #     )
# # #
# # #     X = preprocessor.fit_transform(features)
# # #     y = targets.to_numpy()
# # #
# # #     return X, y
# # #
# # #
# # # def split_data(X, y, test_size=0.1, random_state=0):
# # #     """
# # #     Splits the data into training and testing sets.
# # #
# # #     Parameters:
# # #     - X: Input features.
# # #     - y: Target labels.
# # #     - test_size: Proportion of the dataset to include in the test split.
# # #     - random_state: Controls the shuffling applied to the data before applying the split.
# # #
# # #     Returns:
# # #     - Training and testing sets.
# # #     """
# # #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
# # #
# # #     return np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32), \
# # #         np.array(X_test, dtype=np.float32), np.array(y_test, dtype=np.float32)
# # #
# # #
# # # def predict_and_save(model, X_test, y_test, filename='predictions.csv'):
# # #     # Making predictions
# # #     predictions = model.predict(X_test)
# # #     # Converting predictions from probabilities to binary values (0 or 1)
# # #     predictions_binary = (predictions > 0.5).astype(int)
# # #
# # #     # Combining real and predicted values
# # #     combined = np.hstack((y_test, predictions_binary))  # Concatenate actual and predicted side by side
# # #
# # #     # Creating a DataFrame from the combined array
# # #     columns = ['RealNineToEleven', 'RealElevenToOne', 'RealOneToThree', 'RealThreeToFive',
# # #                'PredictedNineToEleven', 'PredictedElevenToOne', 'PredictedOneToThree', 'PredictedThreeToFive']
# # #     results_df = pd.DataFrame(combined, columns=columns)
# # #
# # #     # Saving to CSV
# # #     results_df.to_csv(filename, index=False)
# # #
# # #     # Evaluating the model
# # #     scores = model.evaluate(X_test, y_test, verbose=0)
# # #     print(f"Test Loss: {scores[0]}")
# # #     print(f"Test Accuracy: {scores[1]}")
# # #
# # #
# # # # def predict_and_save(bst, X_test, y_test, filename='predictions.csv'):
# # # #     dtest = xgb.DMatrix(X_test)
# # # #     y_pred = bst.predict(dtest)
# # # #     predictions_binary = (y_pred > 0.5).astype(int)
# # # #     # Evaluate
# # # #     accuracy = accuracy_score(y_test, predictions_binary)
# # # #     print(f"Test Accuracy: {accuracy}")
# # # #
# # # #     # Combine and save predictions
# # # #     combined = np.hstack((y_test, predictions_binary))
# # # #     columns = ['RealNineToEleven', 'RealElevenToOne', 'RealOneToThree', 'RealThreeToFive',
# # # #                'PredictedNineToEleven', 'PredictedElevenToOne', 'PredictedOneToThree', 'PredictedThreeToFive']
# # # #     results_df = pd.DataFrame(combined, columns=columns)
# # # #     results_df.to_csv(filename, index=False)
# # #
# # #
# # # def build_model(input_shape):
# # #     model = Sequential([
# # #         Dense(10, input_shape=(input_shape,), activation='relu'),
# # #         # Dense(64, activation='relu'),
# # #         Dense(4)
# # #     ])
# # #
# # #     optimizer = Adam(learning_rate=0.0005)
# # #
# # #     model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
# # #     return model
# # #
# # #
# # # # def train_and_evaluate_model(x_train, y_train, x_test, y_test):
# # # #     # Initialize the model
# # # #     model = XGBClassifier(
# # # #         objective='binary:logistic',
# # # #         eval_metric='logloss',
# # # #         use_label_encoder=False
# # # #     )
# # # #
# # # #     # Fit the model
# # # #     model.fit(x_train, y_train)
# # # #
# # # #     # Predictions
# # # #     y_pred = model.predict(x_test)
# # # #     predictions = [round(value) for value in y_pred]
# # # #
# # # #     # Evaluate predictions
# # # #     accuracy = np.mean(predictions == y_test)
# # # #     print("Accuracy: %.2f%%" % (accuracy * 100.0))
# # # #
# # # #     return model
# # #
# # #
# # # def train_model(model, x_train, y_train, x_val, y_val, batch_size):
# # #     learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
# # #
# # #     history = model.fit(
# # #         x_train, y_train, epochs=300,
# # #         verbose=1,
# # #         validation_data=(x_val, y_val),
# # #         callbacks=[learning_rate_reduction],
# # #         batch_size=batch_size
# # #     )
# # #
# # #     plot_history(history)
# # #
# # #     return history
# # #
# # #
# # # def save_model_to_pickle():
# # #     pass
# # #
# # #
# # # def main():
# # #     print("Loading data...")
# # #     X, y = load_data('preprocessed_meeting_room_data.csv')
# # #     x_train, y_train, x_test, y_test = split_data(X, y)
# # #
# # #     # print(x_train.shape[1])
# # #     print("Building Model...")
# # #     model = build_model(x_train.shape[1])
# # #     print(model.summary())
# # #
# # #     # x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
# # #     # x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
# # #
# # #     # model = build_model((x_train.shape[1], x_train.shape[2]))
# # #     # model = build_model()
# # #
# # #     print("Training model...")
# # #     train_model(model, x_train, y_train, x_test, y_test, batch_size=5)
# # #
# # #     print("Predicting and saving results and testing model...")
# # #     predict_and_save(model, x_test, y_test, 'model_predictions.csv')
# # #
# # #     # print(X)
# # #     # print("\n")
# # #     # print(y)
# # #
# # #
# # # # def main():
# # # #     print("Loading data...")
# # # #     X, y = load_data('preprocessed_meeting_room_data.csv')
# # # #     X_train, y_train, X_test, y_test = split_data(X, y, test_size=0.1, random_state=0)
# # # #
# # # #     print("Training model...")
# # # #     bst = train_and_evaluate(X_train, y_train, X_test, y_test)
# # # #
# # # #     print("Predicting and saving results...")
# # # #     predict_and_save(bst, X_test, y_test, 'model_predictions.csv')
# # #
# # #
# # # if __name__ == "__main__":
# # #     main()
# #
# #
# # import pandas as pd
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.model_selection import train_test_split
# # from sklearn.multioutput import MultiOutputClassifier
# # from sklearn.preprocessing import OneHotEncoder, StandardScaler
# # from sklearn.compose import ColumnTransformer
# # from sklearn.pipeline import Pipeline
# # from xgboost import XGBClassifier
# # from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# #
# #
# # def load_and_preprocess_data(filepath):
# #     df = pd.read_csv(filepath)
# #     # Drop unused features and split into features and targets
# #     features = df.drop(['date', 'attendanceNineToEleven', 'attendanceElevenToOne', 'attendanceOneToThree', 'attendanceThreeToFive'], axis=1)
# #     targets = df[['nineToEleven', 'elevenToOne', 'oneToThree', 'threeToFive']]  # Assuming binary [0, 1] indicating booking status
# #
# #     # Split into train and test sets
# #     X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
# #
# #     return X_train, X_test, y_train, y_test
# #
# # def build_and_train_model(X_train, y_train, X_test, y_test):
# #     # Preprocessing for numerical features
# #     numeric_features = ['capacity', 'day_of_week', 'month', 'week_of_year']
# #     numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
# #
# #     # Preprocessing for categorical features
# #     categorical_features = ['room']
# #     categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
# #
# #     # Combine preprocessing steps
# #     preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
# #                                                    ('cat', categorical_transformer, categorical_features)])
# #
# #     # Define the model pipeline with Logistic Regression
# #     model = Pipeline(steps=[('preprocessor', preprocessor),
# #                             ('classifier', MultiOutputClassifier(LogisticRegression(solver='lbfgs', max_iter=1000)))])
# #
# #     model.fit(X_train, y_train)
# #     y_pred = model.predict(X_test)
# #
# #     # Evaluate the model
# #     print("Accuracy:", accuracy_score(y_test, y_pred, normalize=False) / y_test.size)
# #     print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['nineToEleven', 'elevenToOne', 'oneToThree', 'threeToFive']))
# #
# #     return model
# #
# # def main():
# #     filepath = 'preprocessed_meeting_room_data.csv'
# #     X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)
# #     model = build_and_train_model(X_train, y_train, X_test, y_test)
# #     # Here you can save your model using joblib or pickle for later use in your application
# #
# #
# # if __name__ == "__main__":
# #     main()
# #
#
#
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.svm import SVC
#
# def load_and_preprocess_data(filepath):
#     df = pd.read_csv(filepath)
#     features = df.drop(['date', 'attendanceNineToEleven', 'attendanceElevenToOne', 'attendanceOneToThree', 'attendanceThreeToFive'], axis=1)
#     targets = df[['nineToEleven', 'elevenToOne', 'oneToThree', 'threeToFive']]
#     X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
#     return X_train, X_test, y_train, y_test
#
# def build_pipeline(classifier):
#     numeric_features = ['capacity', 'day_of_week', 'month', 'week_of_year']
#     numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
#     categorical_features = ['room']
#     categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
#     preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])
#     pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
#     return pipeline
#
# def evaluate_model(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     print("Accuracy:", accuracy_score(y_test, y_pred))
#     print("Classification Report:\n", classification_report(y_test, y_pred))
#
# def main():
#     filepath = 'preprocessed_meeting_room_data.csv'
#     X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)
#
#     classifiers = {
#         "Logistic Regression": MultiOutputClassifier(LogisticRegression(solver='lbfgs', max_iter=1000)),
#         "Random Forest": MultiOutputClassifier(RandomForestClassifier(n_estimators=100)),
#         "XGBoost": MultiOutputClassifier(XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
#         "SVM": MultiOutputClassifier(SVC(probability=True))
#     }
#
#     for name, classifier in classifiers.items():
#         print(f"Training {name}...")
#         pipeline = build_pipeline(classifier)
#         pipeline.fit(X_train, y_train)
#         print(f"Evaluating {name}...")
#         evaluate_model(pipeline, X_test, y_test)
#
# if __name__ == "__main__":
#     main()
#


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LogisticRegression
# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.metrics import accuracy_score, classification_report
#
#
# def load_and_preprocess_data(filepath):
#     df = pd.read_csv(filepath)
#     features = df.drop(
#         ['date', 'attendanceNineToEleven', 'attendanceElevenToOne', 'attendanceOneToThree', 'attendanceThreeToFive'],
#         axis=1)
#     targets = df[['nineToEleven', 'elevenToOne', 'oneToThree', 'threeToFive']]
#     X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
#     return X_train, X_test, y_train, y_test
#
#
# def build_pipeline():
#     numeric_features = ['capacity', 'day_of_week', 'month', 'week_of_year']
#     numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
#     categorical_features = ['room']
#     categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
#     preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
#                                                    ('cat', categorical_transformer, categorical_features)])
#     pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                                (
#                                'classifier', MultiOutputClassifier(LogisticRegression(solver='lbfgs', max_iter=1000)))])
#     return pipeline
#
#
# def evaluate_and_save_predictions(model, X_test, y_test, filename='predictions.csv'):
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print("Accuracy:", accuracy)
#     print("Classification Report:\n", classification_report(y_test, y_pred))
#
#     # Combine real values and predictions for comparison
#     comparison_df = pd.concat([y_test.reset_index(drop=True), pd.DataFrame(y_pred, columns=['Pred_nineToEleven',
#                                                                                             'Pred_elevenToOne',
#                                                                                             'Pred_oneToThree',
#                                                                                             'Pred_threeToFive'])],
#                               axis=1)
#     comparison_df.to_csv(filename, index=False)
#     print(f"Predictions saved to {filename}")
#
#
# def main():
#     filepath = 'preprocessed_meeting_room_data.csv'  # Make sure to update this path to your actual file location
#     X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)
#
#     model_pipeline = build_pipeline()
#     print("Training Logistic Regression...")
#     model_pipeline.fit(X_train, y_train)
#
#     print("Evaluating Logistic Regression...")
#     evaluate_and_save_predictions(model_pipeline, X_test, y_test, 'logistic_regression_predictions.csv')
#
#
# if __name__ == "__main__":
#     main()



