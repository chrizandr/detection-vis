"""Main application."""
import os
import pdb

from flask import Flask
from flask import (
    request, render_template,
    redirect, url_for, flash, abort
)


def get_info(data_folder):
    models = os.listdir(data_folder)
    valid = [x for x in models if os.path.exists(data_folder, "model.info")]
    return valid


app = Flask(__name__)
print("Configuring app...")
app.secret_key = "DFDKFNWEFOWEFIWV"
app.config["MONGO_URI"] = "mongodb://localhost:27017/demodb"
data_folder = "static/output"
metrics = ["all", "ap50", "ap75", "ap50to95"]
models = get_info(data_folder)


@app.route("/", methods=["GET"])
@app.route("/<string:model>/", methods=["GET"])
@app.route("/<string:model>/<int:metric>", methods=["GET"])
@app.route("/<string:model>/<int:metric>/<float:threshup>", methods=["GET"])
@app.route("/<string:model>/<int:metric>/<float:threshup>/", methods=["GET"])
def index(model="yolo", metric="all", threshup=1, threshlow=0):
    """Index page."""
    if metric not in metrics:
        abort(404)
    if threshup > 1 or threshlow < 0:
        abort(404)

    files, map, dist = get_files_metrics(model, metric, threshup, threshlow)

    return render_template("index.html", files=files, map=map, metric=metric,
                           threshup=threshup, threshlow=threshlow, dist=dist)





if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
