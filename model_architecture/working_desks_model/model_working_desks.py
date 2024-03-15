import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers.legacy import Adam
from keras.src.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

np.set_printoptions(precision=2)


def plot_history(history):
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


def load_data(filepath):
    df = pd.read_csv(filepath)
    X = df[['day_of_week', 'day_of_month', 'month', 'week_of_year']]  # Use numerical date features
    y = df[['firstHalf', 'secondHalf']]  # Predicting availability in both halves
    return X, y


def split_data(X, y, test_size=0.1, random_state=42):
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def build_model(input_shape):
    model = Sequential([
        Dense(100, input_shape=(input_shape,), activation='relu'),
        # Dense(64, activation='relu'),
        # Dense(128, activation='relu'),
        # Dense(128, activation='relu'),
        # Dense(256, activation='relu'),
        # Dense(256, activation='relu'),
        # Dense(256, activation='relu'),
        # Dense(1096, activation='relu'),
        Dense(2)  # Output layer for binary classification of two features
    ])

    optimizer = Adam(learning_rate=0.0005)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, x_train, y_train, x_val, y_val, batch_size=32):
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
    history = model.fit(
        x_train, y_train, epochs=150,
        verbose=1,
        validation_data=(x_val, y_val),
        callbacks=[learning_rate_reduction],
        batch_size=batch_size
    )

    plot_history(history)
    return model


def predict_and_save(model, X_test, y_test, filename='predictions.csv'):
    predictions = model.predict(X_test)
    # Round predictions to nearest integer (0 or 1)
    rounded_predictions = np.round(predictions)
    # Combine real and predicted values
    combined = np.hstack((y_test.reset_index(drop=True), rounded_predictions))
    # Create DataFrame and save to CSV
    predictions_df = pd.DataFrame(combined, columns=['RealFirstHalf', 'RealSecondHalf',
                                                     'PredictedFirstHalf', 'PredictedSecondHalf'])
    predictions_df.to_csv(filename, index=False)


def save_model_to_pickle(model, filename='model.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def main():
    filepath = 'preprocessed_working_desks_data.csv'
    X, y = load_data(filepath)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(X_train)
    model = build_model(X_train.shape[1])
    trained_model = train_model(model, X_train, y_train, X_test, y_test, batch_size=20)
    predict_and_save(trained_model, X_test, y_test, 'desk_availability_predictions.csv')
    save_model_to_pickle(trained_model)


if __name__ == "__main__":
    main()


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# import pickle
#
# np.set_printoptions(precision=2)
#
#
# def load_data(filepath):
#     df = pd.read_csv(filepath)
#     X = df[['day_of_week', 'day_of_month', 'month', 'week_of_year']].values
#     y = df[['firstHalf', 'secondHalf']].values
#     return X, y
#
#
# def split_data(X, y, test_size=0.1, random_state=42):
#     return train_test_split(X, y, test_size=test_size, random_state=random_state)
#
#
# def train_model(X_train, y_train):
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#     return model
#
#
# def predict_and_save(model, X_test, y_test, filename='predictions.csv'):
#     predictions = model.predict(X_test)
#     accuracy = accuracy_score(y_test, predictions)
#     print(f"Test Accuracy: {accuracy}")
#
#     combined = np.hstack((y_test, predictions))
#     predictions_df = pd.DataFrame(combined, columns=['RealFirstHalf', 'RealSecondHalf', 'PredictedFirstHalf',
#                                                      'PredictedSecondHalf'])
#     predictions_df.to_csv(filename, index=False)
#
#
# def save_model_to_pickle(model, filename='model.pkl'):
#     with open(filename, 'wb') as file:
#         pickle.dump(model, file)
#
#
# def main():
#     filepath = 'preprocessed_working_desks_data.csv'
#     X, y = load_data(filepath)
#     X_train, X_test, y_train, y_test = split_data(X, y)
#     model = train_model(X_train, y_train)
#     predict_and_save(model, X_test, y_test, 'desk_availability_predictions.csv')
#     save_model_to_pickle(model, 'rf_desk_availability_model.pkl')
#
#
# if __name__ == "__main__":
#     main()
