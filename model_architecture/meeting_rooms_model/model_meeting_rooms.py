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
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder

np.set_printoptions(precision=2)


def plot_history(history):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def load_data(filepath):
    df = pd.read_csv(filepath)

    # Preprocessing steps
    features = df.drop(
        ['date', 'attendanceNineToEleven', 'attendanceElevenToOne', 'attendanceOneToThree', 'attendanceThreeToFive'],
        axis=1)
    targets = df[['nineToEleven', 'elevenToOne', 'oneToThree', 'threeToFive']]

    # Define numerical and categorical columns
    num_cols = ['capacity', 'day_of_week', 'month', 'week_of_year']
    cat_cols = ['room']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(), cat_cols)
        ]
    )

    X = preprocessor.fit_transform(features)
    y = targets.to_numpy()

    return X, y


def split_data(X, y, test_size=0.1, random_state=0):
    """
    Splits the data into training and testing sets.

    Parameters:
    - X: Input features.
    - y: Target labels.
    - test_size: Proportion of the dataset to include in the test split.
    - random_state: Controls the shuffling applied to the data before applying the split.

    Returns:
    - Training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32), \
        np.array(X_test, dtype=np.float32), np.array(y_test, dtype=np.float32)


# def predict_and_save(model, X_test, y_test, filename='predictions.csv'):
#     # Making predictions
#     predictions = model.predict(X_test)
#     # Converting predictions from probabilities to binary values (0 or 1)
#     predictions_binary = (predictions > 0.5).astype(int)
#
#     # Saving predictions to CSV
#     np.savetxt(filename, predictions_binary, delimiter=",", fmt='%d',
#                header="nineToEleven,elevenToOne,oneToThree,threeToFive", comments="")
#
#     # Evaluating the model
#     scores = model.evaluate(X_test, y_test, verbose=0)
#     print(f"Test Loss: {scores[0]}")
#     print(f"Test Accuracy: {scores[1]}")


# def predict_and_save(model, X_test, y_test, original_data, filename='predictions_comparison.csv'):
#     # Making predictions
#     predictions = model.predict(X_test)
#     # Converting predictions from probabilities to binary values (0 or 1)
#     predictions_binary = (predictions > 0.5).astype(int)
#
#     # Preparing the DataFrame for comparison
#     original_columns = ['RealNineToEleven', 'RealElevenToOne', 'RealOneToThree', 'RealThreeToFive']
#     predicted_columns = ['PredictedNineToEleven', 'PredictedElevenToOne', 'PredictedOneToThree', 'PredictedThreeToFive']
#
#     # Extracting original time slot availability
#     real_data_df = pd.DataFrame(y_test, columns=original_columns)
#     # Creating a DataFrame for predicted availability
#     predicted_data_df = pd.DataFrame(predictions_binary, columns=predicted_columns)
#
#     # Concatenating real and predicted data side by side for comparison
#     comparison_df = pd.concat([real_data_df, predicted_data_df], axis=1)
#
#     # Including the 'room' and 'date' columns from the original dataset for reference
#     comparison_df = pd.concat([original_data[['room', 'date']].reset_index(drop=True), comparison_df], axis=1)
#
#     # Saving the comparison DataFrame to CSV
#     comparison_df.to_csv(filename, index=False)
#
#     # Evaluating the model
#     scores = model.evaluate(X_test, y_test, verbose=0)
#     print(f"Test Loss: {scores[0]}")
#     print(f"Test Accuracy: {scores[1]}")


def predict_and_save(model, X_test, y_test, filename='predictions.csv'):
    # Making predictions
    predictions = model.predict(X_test)
    # Converting predictions from probabilities to binary values (0 or 1)
    predictions_binary = (predictions > 0.5).astype(int)

    # Combining real and predicted values
    combined = np.hstack((y_test, predictions_binary))  # Concatenate actual and predicted side by side

    # Creating a DataFrame from the combined array
    columns = ['RealNineToEleven', 'RealElevenToOne', 'RealOneToThree', 'RealThreeToFive',
               'PredictedNineToEleven', 'PredictedElevenToOne', 'PredictedOneToThree', 'PredictedThreeToFive']
    results_df = pd.DataFrame(combined, columns=columns)

    # Saving to CSV
    results_df.to_csv(filename, index=False)

    # Evaluating the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {scores[0]}")
    print(f"Test Accuracy: {scores[1]}")


def build_model(input_shape):
    model = Sequential([
        Dense(100, input_shape=(input_shape,), activation='relu'),
        Dense(64, activation='relu'),
        Dense(4)
    ])

    optimizer = Adam(learning_rate=0.0005)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, x_train, y_train, x_val, y_val, batch_size):
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

    history = model.fit(
        x_train, y_train, epochs=300,
        verbose=1,
        validation_data=(x_val, y_val),
        callbacks=[learning_rate_reduction],
        batch_size=batch_size
    )

    plot_history(history)

    return history


def save_model_to_pickle():
    pass


def main():
    print("Loading data...")
    X, y = load_data('preprocessed_meeting_room_data.csv')
    x_train, y_train, x_test, y_test = split_data(X, y)

    # print(x_train.shape[1])
    print("Building Model...")
    model = build_model(x_train.shape[1])
    print(model.summary())

    print("Training model...")
    train_model(model, x_train, y_train, x_test, y_test, batch_size=5)

    print("Predicting and saving results and testing model...")
    predict_and_save(model, x_test, y_test, 'model_predictions.csv')


    # print(X)
    # print("\n")
    # print(y)


if __name__ == "__main__":
    main()
