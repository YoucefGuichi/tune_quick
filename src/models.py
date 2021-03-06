import matplotlib.pyplot as plt
import mpld3
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU
import logging as logger

# configs
logger.getLogger().setLevel(logger.INFO)
scaler = MinMaxScaler(feature_range=(0, 1))
plt.style.use('fivethirtyeight')


class Model:
    """Model is the base class which has common function between the sub models"""

    def __init__(self, dataset, sequential):
        """
        :param dataset: dataset header must be with the following order (Date,Open,High,Close)
        :param sequential:
        """
        self.dataset = dataset
        self.sequential = sequential
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.predictions = []
        self.r_mse = None
        self.scaled_close_data_values = None
        self.close_data_values = None
        self.training_data_len = None
        self.close_dataframe = None

    def clean_and_prepare_dataset(self):
        """transfer the data to the correct format"""
        self.dataset.drop_duplicates(inplace=True)
        self.dataset.dropna(inplace=True)
        self.dataset["Date"] = pd.to_datetime(self.dataset["Date"])
        self.dataset["Date"] = self.dataset["Date"].dt.date
        self.dataset.set_index("Date", inplace=True)
        logger.info("validation done")

    def split_data(self):
        """split the data into x_train and y_train and scale it"""
        # Create a new dataframe with only the 'Close column
        self.close_dataframe = self.dataset.filter(['Close'])
        self.close_data_values = self.close_dataframe.values

        # Get the number of rows to train the model on
        self.training_data_len = int(np.ceil(len(self.close_data_values) * .95))
        self.scaled_close_data_values = scaler.fit_transform(self.close_data_values)
        train_data = self.scaled_close_data_values[0:int(self.training_data_len), :]

        # Split the data into x_train and y_train data sets
        for i in range(60, len(train_data)):
            self.x_train.append(train_data[i - 60:i, 0])
            self.y_train.append(train_data[i, 0])
        self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)
        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], self.x_train.shape[1], 1))

    def predict(self):
        """calculate the predictions"""
        test_data = self.scaled_close_data_values[self.training_data_len - 60:, :]
        self.y_test = self.close_data_values[self.training_data_len:, :]
        for i in range(60, len(test_data)):
            self.x_test.append(test_data[i - 60:i, 0])
        x_test = np.array(self.x_test)
        self.x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        self.predictions = self.sequential.predict(x_test)
        self.predictions = scaler.inverse_transform(self.predictions)

    def plot_predictions(self, title, x_label, y_label):
        """plot predictions and convert it to html elements"""
        train = self.close_dataframe[:self.training_data_len]
        valid = self.close_dataframe[self.training_data_len:]
        valid['Predictions'] = self.predictions
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.set_title(title, fontsize=15)
        ax.set_xlabel(x_label, fontsize=15)
        ax.set_ylabel(y_label, fontsize=15)
        ax.plot(train['Close'])
        # ax.plot(valid[['Close']])
        ax.plot(valid['Predictions'], linestyle="dashed")
        ax.legend(['Train', 'Predictions'], loc='lower right')
        plt.show()
        chart = mpld3.fig_to_html(fig)
        return chart

    def calculate_rmse(self):
        """calculate root mean square error"""
        self.r_mse = np.sqrt(np.mean(((self.predictions - self.y_test) ** 2)))


class GRUModel(Model):
    def __init__(self, dataset, sequential):
        super().__init__(dataset, sequential)

    def train(self, optimizer: str, loss='mean_squared_error', epochs=20,
              batch_size=32):
        self.sequential.add(GRU(units=50, return_sequences=True, input_shape=(self.x_train.shape[1], 1),
                                activation='tanh'))
        self.sequential.add(Dropout(0.2))
        self.sequential.add(GRU(units=50, return_sequences=True, input_shape=(self.x_train.shape[1], 1),
                                activation='tanh'))
        self.sequential.add(Dropout(0.2))
        self.sequential.add(GRU(units=50, return_sequences=True, input_shape=(self.x_train.shape[1], 1),
                                activation='tanh'))
        self.sequential.add(Dropout(0.2))
        self.sequential.add(GRU(units=50, activation='tanh'))
        self.sequential.add(Dropout(0.2))
        self.sequential.add(Dense(units=1))
        self.sequential.compile(optimizer=optimizer,
                                loss=loss)

        # fitting the model
        self.sequential.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size)


class LSTMModel(Model):

    def __init__(self, dataset, sequential):
        super().__init__(dataset, sequential)

    def train(self, optimizer: str, loss='mean_squared_error', epochs=1, batch_size=32):
        self.sequential.add(LSTM(units=50, return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
        self.sequential.add(Dropout(0.2))
        self.sequential.add(LSTM(units=50, return_sequences=True))
        self.sequential.add(Dropout(0.2))
        self.sequential.add(LSTM(units=50, return_sequences=True))
        self.sequential.add(Dropout(0.2))
        self.sequential.add(LSTM(units=50))
        self.sequential.add(Dropout(0.2))
        self.sequential.add(Dense(units=1))
        self.sequential.compile(optimizer=optimizer, loss=loss)
        self.sequential.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size)
