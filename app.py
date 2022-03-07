import io

import pandas as pd
from flask import Flask, render_template, request
from tensorflow.keras import Sequential

from src.models import GRUModel

app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        file = request.files["csv-file"]

        if not file:
            return "no file"
        # read csv file
        dataset = pd.read_csv(io.StringIO(file.stream.read().decode("UTF8"), newline=None))
        seq = Sequential()
        model = GRUModel(dataset, seq)
        model.clean_and_prepare_dataset()
        model.split_data()
        model.train(epochs=1)
        model.predict()
        chart = model.plot_predictions("IBM STOCK", "Dates", "Price")
        return render_template("index.html", bar_chart=chart)
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
