import asyncio
import io
import logging as logger

import pandas as pd
from tensorflow.keras import Sequential

from src.models import LSTMModel, GRUModel


class Parser:

    def __init__(self, request):
        self.request = request
        self.chart = None
        self.algorithms = {
            'LSTM': LSTMModel,
            'GRU': GRUModel
        }

    async def add_model_with_dataset_to_queue(self, queue):
        try:
            if self.request.method == "POST":
                dataset = pd.read_csv(
                    io.StringIO(self.request.files["csv-file"].stream.read().decode("UTF8"), newline=None))
                seq = Sequential()
                model = self.algorithms[self.request.form["algorithm"]](dataset, seq)
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
            self.chart = model.plot_predictions(self.request.form["chart-title"], self.request.form["x-axis"],
                                                self.request.form["y-axis"])

    async def main(self):

        queue = asyncio.Queue()
        task1 = asyncio.create_task(self.add_model_with_dataset_to_queue(queue))
        task2 = asyncio.create_task(self.run_model(queue))
        await asyncio.gather(*[
            task1, task2]
        )
