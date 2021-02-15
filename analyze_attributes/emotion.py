import argparse
import glob
import pathlib
import dlib
import numpy as np
import paz.processors as pr
from paz.abstract import Box2D
from paz.backend.image import load_image
from paz.datasets.utils import get_class_names
from paz.applications import MiniXceptionFER
from typing import Dict, List


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


def predict(target_dir: pathlib.Path) -> Dict[str, np.ndarray]:
    results = {}
    detect = EmotionDetector()
    for i, img_file in enumerate(glob.glob(str(target_dir / "*.png"))):
        image = load_image(img_file)
        predictions = detect(image)
        if len(predictions) != 1:
            continue

        print(f"{i:05d} {img_file}", predictions[0][0].tolist())
        results[str(pathlib.Path(img_file).resolve())] = predictions[0][0]

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir", type=pathlib.Path)
    parser.add_argument(
        "--out_file", type=pathlib.Path, default=pathlib.Path("emotions.npz")
    )
    args = parser.parse_args()

    results = predict(args.target_dir.resolve(strict=True))
    np.savez(args.out_file.resolve(), **results)
