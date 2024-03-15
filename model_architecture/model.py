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
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)


def plot_history(history):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend()
    plt.show()


def load_data(input_path_desks, input_path_meeting_rooms):
    input_path_desks = pd.read_csv(input_path_desks, header=None)
    input_path_meeting_rooms = pd.read_csv(input_path_meeting_rooms, header=None)
    return input_path_desks, input_path_meeting_rooms


def split_data():
    pass


def build_model():
    pass


def train_model():
    pass


def save_model_to_pickle():
    pass


def main():
    print("Loading data...")
    input_path_desks, input_path_meeting_rooms = load_data("hackathon-schema.csv", "meeting-rooms.csv")
    # print(input_path_desks.head())
    print(input_path_meeting_rooms.head())
    print("Data loaded successfully.")


if __name__ == "__main__":
    main()
