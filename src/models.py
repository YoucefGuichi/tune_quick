# import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import warnings

plt.style.use("fivethirtyeight")
warnings.filterwarnings("ignore")


class LSTMPredictor:

    def __init__(self, dataset, sequential):
        self.dataset = dataset
        self.sequential = sequential

    def clean_dataset(self):
        self.dataset.drop_duplicates(inplace=True)
        self.dataset.dropna(inplace=True)
        print("validation done")

    def split_data(self, train_end_date, test_start_date):
        train = self.dataset[:'2016'].iloc[:, 1:2].values
        test = self.dataset['2017':].iloc[:, 1:2].values
        sc = MinMaxScaler(feature_range=(0, 1))
        train_scaled = sc.fit_transform(train)

        x_train = []
        y_train = []

        for i in range(60, 2769):
            x_train.append(train_scaled[i - 60:i, 0])
            y_train.append(train_scaled[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        return x_train, y_train, test

    def train(self, x_train: list, y_train: list):
        self.sequential.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        self.sequential.add(Dropout(0.2))
        self.sequential.add(LSTM(units=50, return_sequences=True))
        self.sequential.add(Dropout(0.2))
        self.sequential.add(LSTM(units=50, return_sequences=True))
        self.sequential.add(Dropout(0.2))
        self.sequential.add(LSTM(units=50))
        self.sequential.add(Dropout(0.2))
        self.sequential.add(Dense(units=1))
        self.sequential.compile(optimizer='rmsprop', loss='mean_squared_error')
        self.sequential.fit(x_train, y_train, epochs=5, batch_size=32)
        self.sequential.compile(optimizer='rmsprop', loss='mean_squared_error')
        self.sequential.fit(x_train, y_train, epochs=5, batch_size=32)

    def predict(self, test_data):
        dataset_total = pd.concat((self.dataset['High'][:'2016'], self.dataset['High']['2017':]), axis=0)
        inputs = dataset_total[len(self.dataset) - len(test_data) - 60:].values
        inputs = inputs.reshape(-1, 1)
        sc = MinMaxScaler(feature_range=(0, 1))
        inputs = sc.fit_transform(inputs)
        x_test = []
        for i in range(60, 311):
            x_test.append(inputs[i - 60:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        predicted = self.sequential.predict(x_test)
        predicted = sc.inverse_transform(predicted)

        return predicted
