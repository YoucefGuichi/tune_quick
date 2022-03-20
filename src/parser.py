import asyncio
import io
import logging as logger

import pandas as pd
from flask import render_template

from src.models import LSTMModel, GRUModel
from tensorflow.keras import Sequential

# algorithms dict
algorithms = {
    'LSTM': LSTMModel,
    'GRU': GRUModel
}


class Parser:

    def __init__(self, request):
        self.request = request

    async def add_model_with_dataset_to_queue(self, queue):
        try:
            dataset = await pd.read_csv(
                io.StringIO(self.request.file["csv-file"].stream.read().decode("UTF8"), newline=None))
            seq = Sequential()
            model = algorithms[self.request.form["algorithm"]](dataset, seq)
            queue.put_nowait(model)
        except Exception as err:
            raise err

    async def run_model(self, queue):
        if self.request.method == "POST":
            # get model from the queue
            model = await queue.get()
            logger.info('cleaning ans preparing the data...')
            try:
                model.clean_and_prepare_dataset()
                model.split_data()
            except Exception as err:
                raise err

            logger.info('training started...')
            model.train(
                optimizer=self.request.form["optimizer"],
                loss=self.request.form["loss-function"],
                epochs=int(self.request.form["epochs"]),
                batch_size=int(self.request.form["batch-size"]))
            logger.info('training done')
            model.predict()
            chart = model.plot_predictions(self.request.form["chart-title"], self.request.form["x-axis"],
                                           self.request.form["y-axis"])
            return render_template("index.html", bar_chart=chart)

        else:
            return render_template("index.html")

    async def main(self, model):

        queue = asyncio.Queue
        await asyncio.wait([
            self.add_model_with_dataset_to_queue(queue),
            self.run_model(queue)]
        )
