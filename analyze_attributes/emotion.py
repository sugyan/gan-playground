import argparse
import glob
import os
import dlib
import numpy as np
import pandas as pd
import paz.processors as pr
from paz.abstract import Box2D
from paz.backend.image import load_image
from paz.datasets.utils import get_class_names
from paz.applications import MiniXceptionFER

emotions = get_class_names("FER")


class EmotionDetector(pr.Processor):
    def __init__(self):
        super(EmotionDetector, self).__init__()
        self.detector = dlib.get_frontal_face_detector()
        self.crop = pr.CropBoxes2D()
        self.classify = MiniXceptionFER()

    def call(self, image):
        detections, scores, _ = self.detector.run(image, 1)
        boxes2D = []
        for detection, score in zip(detections, scores):
            boxes2D.append(
                Box2D(
                    [
                        detection.left(),
                        detection.top(),
                        detection.right(),
                        detection.bottom(),
                    ],
                    score,
                )
            )
        results = []
        for cropped_image in self.crop(image, boxes2D):
            results.append(self.classify(cropped_image)["scores"])
        return results


def predict(target_dir, data_file):
    detect = EmotionDetector()
    indexes = []
    columns = []
    for i, img_file in enumerate(glob.glob(os.path.join(target_dir, "*.png"))):
        image = load_image(img_file)
        predictions = detect(image)
        if len(predictions) != 1:
            continue

        print(f"{i:05d} {img_file}", predictions[0][0].tolist())
        indexes.append(os.path.abspath(img_file))
        columns.append(predictions[0][0])

    df = pd.DataFrame(columns, index=indexes, columns=emotions)
    print(df)
    df.to_hdf(data_file, key="df")


def calc_vectors(df, out_file):
    all_dlatents = []
    for index in df.index:
        all_dlatents.append(np.load(f"{index}.npy"))

    k = round(len(df) / 100.0)
    outputs = {}
    for e in emotions:
        if e == "neutral":
            continue
        print(f"top {k} images of {e}")
        dlatents = []
        for index, row in df.sort_values(e, ascending=False)[:k].iterrows():
            print(index, row[e])
            dlatents.append(np.load(f"{index}.npy"))
        outputs[e] = np.mean(dlatents, axis=0) - np.mean(all_dlatents, axis=0)
    np.savez(out_file, **outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir", type=str)
    parser.add_argument("--data_file", type=str, default="emotions.h5")
    parser.add_argument("--out_file", type=str, default="emotions.npz")
    args = parser.parse_args()

    if not os.path.exists(args.data_file):
        predict(args.target_dir, args.data_file)

    df = pd.read_hdf(args.data_file)
    calc_vectors(df, args.out_file)
