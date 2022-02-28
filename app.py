import csv
import io

import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        file = request.files["csv-file"]
        print(file)
        if not file:
            return "no file"
        # read file
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        csv_input = csv.reader(stream)

        # convert the file to a dataframe
        dataset = pd.DataFrame(csv_input)
        print(dataset)

        return render_template("index.html")
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
