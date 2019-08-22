"""Main application."""
import os
import pdb
import io
import base64
import numpy as np
from flask import Flask
from flask import (
    request, render_template,
    url_for, abort
)
import matplotlib.pyplot as plt


def get_info(data_folder):
    models = os.listdir(data_folder)
    valid, labels = [], []
    for m in models:
        model_file = os.path.join(data_folder, m, "model.info")
        if os.path.exists(model_file):
            valid.append(m)
            labels.append(open(model_file, "r").read().split('\n')[0])

    return valid, labels


app = Flask(__name__)
print("Configuring app...")
app.secret_key = "DFDKFNWEFOWEFIWV"
app.config["MONGO_URI"] = "mongodb://localhost:27017/demodb"
data_folder = "static/output"
metrics = ["all", "ap50", "ap75", "ap50to95"]
metric_labels = ["All", "AP@50", "AP@75", "AP@[0.5:0.95]"]
models, labels = get_info(data_folder)


@app.route("/", methods=["GET"])
def index():
    """Index page."""
    model = request.args.get("model", "yolo")
    metric = request.args.get("metric", "all")
    try:
        threshup = float(request.args.get("threshup", 1))
        threshlow = float(request.args.get("threshlow", 0))
    except ValueError:
        abort(404)

    if metric not in metrics:
        abort(404)
    if threshup > 1 or threshlow < 0:
        abort(404)
    if model not in models:
        abort(404)

    files, map, map_label, graph = get_files_metrics(model, metric, threshup, threshlow)
    label = labels[models.index(model)]
    # pdb.set_trace()
    return render_template("index.html", files=files, map=map, metric=metric,
                           threshup=threshup, threshlow=threshlow, model=model,
                           label=label, labels=labels, models=models, metrics=metrics,
                           graph=graph, map_label=map_label, metric_labels=metric_labels)


def get_files_metrics(model, metric, threshup, threshlow):
    f = open(os.path.join(data_folder, model, "model.info"), "r")
    detected_files = os.listdir(os.path.join(data_folder, model))

    files = []
    aps = []
    for i, line in enumerate(f):
        if i == 0:
            continue
        data = line.split()
        files.append(data[0])
        aps.append([float(x) for x in data[1::]])
    aps = np.array(aps)

    ap_50 = aps[:, 0]
    indices_50 = ((ap_50 >= threshlow) & (ap_50 <= threshup)).nonzero()[0]
    map_50 = ap_50[indices_50].mean()
    dist_50, bins_50 = np.histogram(ap_50[indices_50], range=(threshlow, threshup))
    if len(indices_50) == 0:
        ap_50 = np.zeros(len(ap_50))
        map_50, dist_50, bins_50 = 0, [], []

    ap_75 = aps[:, 5]
    indices_75 = ((ap_75 >= threshlow) & (ap_75 <= threshup)).nonzero()[0]
    map_75 = ap_75[indices_75].mean()
    dist_75, bins_75 = np.histogram(ap_75[indices_75], range=(threshlow, threshup))
    if len(indices_75) == 0:
        ap_75 = np.zeros(len(ap_75))
        map_75, dist_75, bins_75 = 0, [], []

    ap_50to95 = np.mean(aps, axis=1)
    indices_50to95 = ((ap_50to95 >= threshlow) & (ap_50to95 <= threshup)).nonzero()[0]
    map_50to95 = ap_50to95[indices_50to95].mean()
    dist_50to95, bins_50to95 = np.histogram(ap_50to95[indices_50to95], range=(threshlow, threshup))
    if len(indices_50to95) == 0:
        ap_50to95 = np.zeros(len(ap_50to95))
        map_50to95, dist_50to95, bins_50to95 = 0, [], []

    outfiles = []
    map_label = None
    fig = plt.figure()
    ax = fig.add_axes([1, 1, 1, 1])
    if metric == "ap50" or metric == "all":
        map_label = "mAP50"
        map = map_50
        outfiles.extend([files[x] for x in indices_50 if files[x] in detected_files])
        ax.plot(bins_50[0:-1], dist_50, color='red', label="AP50 distribution")
    if metric == "ap75" or metric == "all":
        map_label = "mAP75"
        map = map_75
        outfiles.extend([files[x] for x in indices_75 if files[x] in detected_files])
        ax.plot(bins_75[0:-1], dist_75, color='green', label="AP75 distribution")
    if metric == "ap50to95" or metric == "all":
        map_label = "mAP@[0.5:0.95]"
        map = map_50to95
        outfiles.extend([files[x] for x in indices_50to95 if files[x] in detected_files])
        ax.plot(bins_50to95[0:-1], dist_50to95, color='blue', label="AP@[50:95] distribution")
    if metric == "all":
        map = [map_50, map_75, map_50to95]

    pdb.set_trace()
    ax.set_title("Distribution of Average precision based on metric and thresholds")
    ax.set_xlabel("Average Precision")
    ax.set_ylabel("Frequency")
    ax.grid()
    ax.legend()
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    graph = 'data:image/png;base64,{}'.format(graph_url)
    return list(set(outfiles)), map, map_label, graph


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8888, debug=True)
