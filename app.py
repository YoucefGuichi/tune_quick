import io
import logging as logger

import pandas as pd
from flask import Flask, render_template, request
from tensorflow.keras import Sequential

from src.models import GRUModel, LSTMModel

app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        file = request.files["csv-file"]
        algorithm = request.form["algorithm"]
        loss_function = request.form["loss-function"]
        epochs = request.form["epochs"]
        batch_size = request.form["batch-size"]
        optimizer = request.form["optimizer"]
        chart_title = request.form["chart-title"]
        x_axis_title = request.form["x-axis"]
        y_axis_title = request.form["y-axis"]

        # read csv file
        try:
            dataset = pd.read_csv(io.StringIO(file.stream.read().decode("UTF8"), newline=None))
        except Exception as err:
            raise err

        seq = Sequential()
        if algorithm == 'LSTM':
            model = LSTMModel(dataset, seq)
        elif algorithm == 'GRU':
            model = GRUModel(dataset, seq)

        logger.info('cleaning ans preparing the data...')
        try:
            model.clean_and_prepare_dataset()
            model.split_data()
        except Exception as err:
            raise err

        logger.info('training started...')
        model.train(optimizer=optimizer, loss=loss_function, epochs=int(epochs), batch_size=int(batch_size))
        logger.info('training done')
        model.predict()
        chart = model.plot_predictions(chart_title, x_axis_title, y_axis_title)
        return render_template("index.html", bar_chart=chart)
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
