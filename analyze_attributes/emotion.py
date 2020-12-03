import argparse
import glob
import os
import numpy as np
import pandas as pd
import paz.processors as pr
from paz.backend.image import load_image
from paz.datasets.utils import get_class_names
from paz.applications import HaarCascadeFrontalFace, MiniXceptionFER

emotions = get_class_names("FER")


class EmotionDetector(pr.Processor):
    def __init__(self):
        super(EmotionDetector, self).__init__()
        self.detect = HaarCascadeFrontalFace(draw=False)
        self.crop = pr.CropBoxes2D()
        self.classify = MiniXceptionFER()

    def call(self, image):
        boxes2D = self.detect(image)["boxes2D"]
        cropped_images = self.crop(image, boxes2D)
        results = []
        for cropped_image, box2D in zip(cropped_images, boxes2D):
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


def top_k(df, outdir, k=10):
    all_dlatents = []
    for index in df.index:
        all_dlatents.append(np.load(f"{index}.npy"))

    for e in emotions:
        # TODO
        if e != "happy":
            continue

        print(f"top {k} images of {e}")
        dlatents = []
        for index, row in df.sort_values(e, ascending=False)[:k].iterrows():
            print(index, row[e])
            dlatents.append(np.load(f"{index}.npy"))
        np.save(
            os.path.join(outdir, f"{e}.npy"),
            np.mean(dlatents, axis=0) - np.mean(all_dlatents, axis=0),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir", type=str)
    parser.add_argument("--data_file", type=str, default="emotion.h5")
    parser.add_argument("--outdir", type=str, default="out")
    args = parser.parse_args()

    if not os.path.exists(args.data_file):
        predict(args.target_dir, args.data_file)

    df = pd.read_hdf(args.data_file)
    top_k(df, args.outdir, 30)
