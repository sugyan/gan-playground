import argparse
import glob
import dlib
import numpy as np
import pandas as pd
import pathlib
import paz.processors as pr
from paz.abstract import Box2D
from paz.backend.image import load_image
from paz.datasets.utils import get_class_names
from paz.applications import MiniXceptionFER
from typing import List


emotions = get_class_names("FER")


class EmotionDetector(pr.Processor):  # type: ignore
    def __init__(self) -> None:
        super(EmotionDetector, self).__init__()
        self.detector = dlib.get_frontal_face_detector()
        self.crop = pr.CropBoxes2D()
        self.classify = MiniXceptionFER()

    def call(self, image: np.ndarray) -> List[np.ndarray]:
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


def predict(target_dir: pathlib.Path) -> pd.DataFrame:
    detect = EmotionDetector()
    indexes = []
    columns = []
    for i, img_file in enumerate(glob.glob(str(target_dir / "*.png"))):
        image = load_image(img_file)
        predictions = detect(image)
        if len(predictions) != 1:
            continue

        print(f"{i:05d} {img_file}", predictions[0][0].tolist())
        indexes.append(pathlib.Path(img_file).resolve())
        columns.append(predictions[0][0])

    return pd.DataFrame(columns, index=indexes, columns=emotions)


def calc_vectors(df: pd.DataFrame, out_file: pathlib.Path) -> None:
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
    parser.add_argument("target_dir", type=pathlib.Path)
    parser.add_argument(
        "--data_file", type=pathlib.Path, default=pathlib.Path("emotions.h5")
    )
    parser.add_argument(
        "--out_file", type=pathlib.Path, default=pathlib.Path("emotions.npz")
    )
    args = parser.parse_args()

    if args.data_file.exists():
        df = pd.read_hdf(args.data_file.resolve(strict=True))
    else:
        df = predict(args.target_dir.resolve(strict=True))
        print(df)
        df.to_hdf(args.data_file.resolve(), key="df")
    calc_vectors(df, args.out_file.resolve(strict=True))
