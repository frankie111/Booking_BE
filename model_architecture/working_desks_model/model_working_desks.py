import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers.legacy import Adam
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.src.layers import Flatten, Dropout, BatchNormalization, Reshape, LSTM
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
    X = df[['day_of_week', 'day_of_month', 'month', 'week_of_year']].values  # Use numerical date features
    y = df[['firstHalf', 'secondHalf']].values  # Predicting availability in both halves
    return X, y


def split_data(X, y, test_size=0.1, random_state=42):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)


def build_model(input_shape):
    model = Sequential([
        Dense(128, input_shape=(input_shape,), activation='relu'),
        BatchNormalization(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        # Dense(256, activation='relu'),
        BatchNormalization(),
        Dense(2, activation='sigmoid')
    ])

    optimizer = Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model




def train_model(model, x_train, y_train, x_val, y_val, batch_size):
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
    # early_stop = EarlyStopping(monitor='val_loss', verbose=1, patience=20)


    history = model.fit(
        x_train, y_train, epochs=300,
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
    combined = np.hstack((y_test, rounded_predictions))
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
    print(X)
    print(y)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(X_test)
    print(y_test)
    model = build_model(X_train.shape[1])
    trained_model = train_model(model, X_train, y_train, X_test, y_test, batch_size=100)
    predict_and_save(trained_model, X_test, y_test, 'desk_availability_predictions_ann.csv')
    save_model_to_pickle(trained_model)


if __name__ == "__main__":
    main()








# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import pickle
#
#
# def load_data(filepath):
#     df = pd.read_csv(filepath)
#     X = df[['day_of_week', 'day_of_month', 'month', 'week_of_year']]  # Use numerical date features
#     y = df[['firstHalf']]  # Predicting availability in the first half
#     return X, y
#
#
# def split_data(X, y, test_size=0.1, random_state=42):
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     return train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
#
#
# def train_model(X_train, y_train, classifier):
#     model = classifier
#     model.fit(X_train, y_train.values.ravel())  # Some classifiers expect 1D array for labels
#     return model
#
#
# def evaluate_model(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     report = classification_report(y_test, y_pred)
#     matrix = confusion_matrix(y_test, y_pred)
#     print("Accuracy:", accuracy)
#     print("Classification Report:\n", report)
#     print("Confusion Matrix:\n", matrix)
#     return accuracy
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
#
#     classifiers = {
#         "SVM": SVC(kernel='rbf'),  # Support Vector Machine
#         "Random Forest": RandomForestClassifier(),  # Random Forest Classifier
#         "Gradient Boosting": GradientBoostingClassifier(),  # Gradient Boosting Classifier
#         "KNN": KNeighborsClassifier()  # K-Nearest Neighbors Classifier
#     }
#
#     best_accuracy = 0
#     best_model = None
#     best_classifier = None
#
#     for name, classifier in classifiers.items():
#         print(f"Training {name}...")
#         model = train_model(X_train, y_train, classifier)
#         accuracy = evaluate_model(model, X_test, y_test)
#
#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             best_model = model
#             best_classifier = name
#
#         print("\n")
#
#     print(f"Best classifier: {best_classifier} with accuracy: {best_accuracy}")
#     save_model_to_pickle(best_model)
#
#
# if __name__ == "__main__":
#     main()