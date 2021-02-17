# Use https://github.com/shaoanlu/face_toolbox_keras

import argparse
import glob
import pathlib
import cv2
import numpy as np
from typing import Dict
from models.parser import face_parser


def predict(target_dir: pathlib.Path) -> Dict[str, np.ndarray]:
    fp = face_parser.FaceParser()
    # fd = face_detector.FaceAlignmentDetector()
    # fp.set_detector(fd)
    results = {}
    for i, image_path in enumerate(glob.glob(str(target_dir / "*.png"))):
        print(f"{i:5} {image_path}")
        im = cv2.imread(image_path)[..., ::-1]
        results[image_path] = fp.parse_face(im)[0]

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir", type=pathlib.Path)
    parser.add_argument(
        "--out_file", type=pathlib.Path, default=pathlib.Path("parsing_results.npz")
    )
    args = parser.parse_args()

    results = predict(args.target_dir.resolve(strict=True))
    np.savez(args.out_file.resolve(), **results)
