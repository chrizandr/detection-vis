import os
import numpy as np
import pdb
import xml.etree.ElementTree as ET


def create_info_file(output_dir, data_dir, annotation_dir, size):
    """Evaluate output for the ball data."""
    files = os.listdir(data_dir)
    output_files = os.listdir(output_dir)
    output_files = [x for x in output_files if x.endswith(".txt")]
    output_imgs = [os.path.splitext(x)[0] for x in output_files]
    annotations = get_scaled_annotations_PVOC(annotation_dir, size)
    fwriter = open("model.info", "w")
    for f in files:
        aps = []
        for overlap in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
            annotation = annotations[f]
            if f not in output_imgs:
                aps.append(0)
                # print(f, f not in output_imgs)
            else:
                preds, conf = read_output_file(os.path.join(output_dir, f + ".txt"))
                if len(preds) == 0:
                    aps.append(0)
                else:
                    target_cls = np.zeros(annotation.shape[0])
                    pred_cls = np.zeros(preds.shape[0])
                    tps = match_annotations(preds, annotation, overlap)
                    p, r, ap, f1, _ = ap_per_class(tps, conf, pred_cls, target_cls)
                    aps.append(ap[0])
        s = [f] + [str(x) for x in aps]
        fwriter.write(" ".join(s) + "\n")
    fwriter.close()


def read_output_file(filename):
    """Read the output files for predictions."""
    annotations = []
    confs = []
    with open(filename, "r") as f:
        for line in f:
            d = line.split()
            cords = [int(float(x)) for x in d[0:4]]
            conf = float(d[-1])
            annotations.append(cords)
            confs.append(conf)
    return np.array(annotations), np.array(confs)


def match_annotations(output, annotation, overlap):
    """Match the annotations and output in the image and find accuracy."""
    tps = []
    for o in output:
        # pdb.set_trace()
        dist = np.sum((annotation - o) ** 2, axis=1)
        closest_box = annotation[np.argmin(dist), :]
        iou = bb_intersection_over_union(closest_box, o)
        tps.append(int(iou >= overlap))

    return np.array(tps)


def bb_intersection_over_union(boxA, boxB):
    """Find IOU between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def get_scaled_annotations_PVOC(annotation_dir, new_size=(1024, 1024)):
    """Read and scale annotations based on new image size."""
    files = os.listdir(annotation_dir)
    annotations = dict()
    for f in files:
        try:
            file = ET.parse(os.path.join(annotation_dir, f))
            root = file.getroot()
        except Exception:
            pdb.set_trace()

        as_ = root.findall("object")
        for annotation in as_:
            bbox = annotation.findall("bndbox")[0]
            xmin = int(bbox.findall("xmin")[0].text)
            ymin = int(bbox.findall("ymin")[0].text)
            xmax = int(bbox.findall("xmax")[0].text)
            ymax = int(bbox.findall("ymax")[0].text)

            size = root.findall("size")[0]
            width = int(size.findall("width")[0].text)
            height = int(size.findall("height")[0].text)

            new_h, new_w = new_size
            new_h, new_w = float(new_h), float(new_w)
            ymin = int(ymin/(height/new_h))
            ymax = int(ymax/(height/new_h))
            xmin = int(xmin/(width/new_w))
            xmax = int(xmax/(width/new_w))
            name = f.strip(".xml") + ".png"
            if name in annotations:
                annotations[name] = np.vstack((annotations[name], [xmin, ymin, xmax, ymax]))
            else:
                annotations[name] = np.array([xmin, ymin, xmax, ymax]).reshape(1, 4)
    return annotations


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves."""
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

            # Plot
            # plt.plot(recall_curve, precision_curve)

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves."""
    # correct AP calculation
    # first append sentinel values at the end

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


if __name__ == "__main__":
    output_dir = os.getcwd()
    data_dir = "/home/chris_andrew/sports/detection_exp/annotated/"
    annotation_dir = "/home/chris_andrew/sports/detection_exp/annotations/"

    output_dir = os.getcwd()
    # height, width of output images
    size = (1024, 1024)

    create_info_file(output_dir, data_dir, annotation_dir, size)
