# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from keras import Input, Model
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers.legacy import Adam
# from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
# from keras.src.layers import Flatten, Dropout, BatchNormalization, Reshape, LSTM, Activation
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# import pickle
#
# np.set_printoptions(precision=2)
#
#
# def plot_history(history):
#     plt.plot(history.history['loss'], label='Train')
#     plt.plot(history.history['val_loss'], label='Validation')
#     plt.title('Model Loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend()
#     plt.show()
#
#
# def load_data(filepath):
#     df = pd.read_csv(filepath)
#     X = df[['day_of_week', 'day_of_month', 'month', 'week_of_year']].values  # Use numerical date features
#     y = df[['firstHalf', 'secondHalf']].values  # Predicting availability in both halves
#     return X, y
#
# # def load_data(filepath):
# #     df = pd.read_csv(filepath)
# #
# #     # One-hot encoding 'desk' names
# #     desks_encoded = pd.get_dummies(df['desk'], prefix='desk')
# #     df = pd.concat([df, desks_encoded], axis=1)
# #     df.drop(['desk'], axis=1, inplace=True)
# #
# #     # Assuming 'firstHalf' and 'secondHalf' are already binary (0 and 1)
# #     X = df.drop(['row', 'date', 'firstHalf', 'secondHalf'], axis=1)
# #     y = df[['firstHalf', 'secondHalf']]
# #
# #     return X.values, y.values
#
#
#
#
# def split_data(X, y, test_size=0.1, random_state=42):
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     return train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
#
#
# # def split_data(X, y, desks, test_size=0.1, random_state=42):
# #     X_train, X_test, y_train, y_test, desks_train, desks_test = train_test_split(
# #         X, y, desks, test_size=test_size, random_state=random_state)
# #     return X_train, X_test, y_train, y_test, desks_train, desks_test
#
#
#
# # def build_model(input_shape):
# #     model = Sequential([
# #         Dense(128, input_shape=(input_shape,), activation='relu'),
# #         # BatchNormalization(),
# #         Dense(128, activation='relu'),
# #         # BatchNormalization(),
# #         Dense(256, activation='relu'),
# #         # BatchNormalization(),
# #         # Dense(256, activation='relu'),
# #         # BatchNormalization(),
# #         Dense(2, activation='sigmoid')
# #     ])
# #
# #     optimizer = Adam(learning_rate=0.001)
# #
# #     model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
# #     return model
#
# def build_model(input_shape):
#     model = Sequential([
#         Dense(128, input_shape=(input_shape,), activation='relu'),
#         # BatchNormalization(),
#         # Dropout(0.3),
#         Dense(256, activation='relu'),
#         # Dense(256, activation='relu'),
#         # Dense(256, activation='relu'),
#         # BatchNormalization(),
#         Dense(2, activation='sigmoid')
#     ])
#
#     optimizer = Adam(learning_rate=0.0001)
#
#     model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#     return model
#
#
#
#
#
# def train_model(model, x_train, y_train, x_val, y_val, batch_size):
#     learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
#     # early_stop = EarlyStopping(monitor='val_loss', verbose=1, patience=20)
#
#
#     history = model.fit(
#         x_train, y_train, epochs=100,
#         verbose=1,
#         validation_data=(x_val, y_val),
#         callbacks=[learning_rate_reduction],
#         batch_size=batch_size
#     )
#
#     plot_history(history)
#     return model
#
#
# # def train_model(model, x_train, y_train, x_val, y_val, batch_size):
# #     learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
# #
# #     history = model.fit(
# #         x_train,
# #         {"first_half": y_train['firstHalf'], "second_half": y_train['secondHalf']},
# #                      validation_data=(x_val, {"first_half": y_val['firstHalf'], "second_half": y_val['secondHalf']}),
# #                      batch_size=batch_size, epochs=50, verbose=1, callbacks=learning_rate_reduction)
# #
# #     plot_history(history)
# #
# #     return history
#
# def predict_and_save(model, X_test, y_test, filename='predictions.csv'):
#     predictions = model.predict(X_test)
#     rounded_predictions = np.round(predictions)
#
#     # Add dates to the combined array
#     combined = np.hstack((y_test, rounded_predictions))
#
#     # Modify DataFrame columns to include dates
#     predictions_df = pd.DataFrame(combined, columns=['RealFirstHalf', 'RealSecondHalf',
#                                                      'PredictedFirstHalf', 'PredictedSecondHalf'])
#     predictions_df.to_csv(filename, index=False)
#
# # def predict_and_save(model, X_test, y_test, desks_test, filename='predictions.csv'):
# #     predictions = model.predict(X_test)
# #     rounded_predictions = np.round(predictions)
# #
# #     # Convert desks_test to numpy array before reshaping
# #     desks_test_array = desks_test.to_numpy().reshape(-1, 1)
# #     combined = np.hstack((desks_test_array, y_test, rounded_predictions))
# #     predictions_df = pd.DataFrame(combined, columns=['Desk', 'RealFirstHalf', 'RealSecondHalf',
# #                                                      'PredictedFirstHalf', 'PredictedSecondHalf'])
# #     predictions_df.to_csv(filename, index=False)
#
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
#     print(X)
#     print(y)
#     X_train, X_test, y_train, y_test = split_data(X, y)
#     print(X_test)
#     print(y_test)
#     model = build_model(X_train.shape[1])
#     trained_model = train_model(model, X_train, y_train, X_test, y_test, batch_size=15)
#     predict_and_save(trained_model, X_test, y_test, 'desk_availability_predictions_ann1.csv')
#     save_model_to_pickle(trained_model)
#
#
#
#
# if __name__ == "__main__":
#     main()








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


# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, InputLayer
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
#
# np.set_printoptions(precision=2)
#
# # Function to plot the training history
# def plot_history(history):
#     plt.plot(history.history['loss'], label='train')
#     plt.plot(history.history['val_loss'], label='validation')
#     plt.title('Model loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Validation'], loc='upper left')
#     plt.show()
#
# # Function to load and preprocess data
# def load_data(filepath):
#     data = pd.read_csv(filepath)
#     X = data[['day_of_week', 'day_of_month', 'month', 'week_of_year']].values
#     y = data[['firstHalf', 'secondHalf']].values
#     return X, y
#
# # Function to split data into training and test sets
# def split_data(X, y, test_size=0.1, random_state=0):
#     return train_test_split(X, y, test_size=test_size, random_state=random_state)
#
# # Function to build the neural network model
# def build_model(input_shape):
#     model = Sequential([
#         InputLayer(input_shape=input_shape),
#         Dense(128, activation='relu'),
#         Dense(64, activation='relu'),
#         Dense(2, activation='sigmoid')  # 2 output neurons for firstHalf and secondHalf predictions
#     ])
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model
#
# # Function to train the model
# def train_model(model, x_train, y_train, x_val, y_val, batch_size=32, epochs=100):
#     return model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs)
#
# # Main function to run the model training and evaluation
# def main():
#     filepath = 'preprocessed_working_desks_data.csv'  # Update this path
#     X, y = load_data(filepath)
#     x_train, x_val, y_train, y_val = split_data(X, y)
#     model = build_model(input_shape=(x_train.shape[1],))
#     history = train_model(model, x_train, y_train, x_val, y_val)
#     plot_history(history)
#     # Save the model for later use
#     model.save('desk_availability_model.h5')
#
# if __name__ == "__main__":
#     main()



import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['desk'] = df['desk'].astype('category').cat.codes
    df = df.drop(['date'], axis=1)
    return df

def split_data(df, test_size=0.5, random_state=0):
    X = df.drop(['firstHalf', 'secondHalf'], axis=1)
    y = df[['firstHalf', 'secondHalf']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def build_model():
    model = RandomForestClassifier()
    # parameters = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [None, 10, 20, 30],
    #     'criterion': ['gini', 'entropy']
    # }
    # cv = GridSearchCV(model, parameters, cv=5)
    return model

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def predict_and_save(model, X_test, y_test, filename='predictions.csv'):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy: {accuracy * 100}%')
    output = pd.DataFrame(predictions, columns=['predictedFirstHalf', 'predictedSecondHalf'])
    output['realFirstHalf'] = y_test['firstHalf'].values
    output['realSecondHalf'] = y_test['secondHalf'].values
    output.to_csv(filename, index=False)

def save_model_to_pickle(model, filename='model.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def main():
    df = load_data('preprocessed_working_desks_data.csv')
    X_train, X_test, y_train, y_test = split_data(df)
    model = build_model()
    model = train_model(model, X_train, y_train)
    predict_and_save(model, X_test, y_test, filename='test.csv')
    save_model_to_pickle(model)

if __name__ == "__main__":
    main()