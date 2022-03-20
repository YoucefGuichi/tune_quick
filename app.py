import asyncio

from flask import Flask, render_template, request

from src.parser import Parser

app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def index():
    parser = Parser(request=request)
    asyncio.run(parser.main())
    return render_template("index.html", bar_chart=parser.chart)


if __name__ == "__main__":
    app.run(debug=True)
