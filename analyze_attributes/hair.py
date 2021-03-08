import argparse
import pathlib
from typing import Dict

import cv2
import numpy as np


def calculate(parsing_results: pathlib.Path) -> Dict[str, np.ndarray]:
    results = {}
    data = np.load(parsing_results)

    for i, (k, v) in enumerate(data.items()):
        mask = v == 17
        rows = mask.sum(axis=1)
        lowest = rows.nonzero()[0][-1]
        img = cv2.imread(k, cv2.IMREAD_COLOR)
        img[~mask] = [0, 0, 0]
        result = [
            mask.sum(),
            lowest + rows[lowest] / len(rows),
            img.sum() / mask.sum(),
        ]
        print(f"{i:05d} {k}", result)
        results[k] = result
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parsing_results",
        type=pathlib.Path,
        default=pathlib.Path("parsing_results.npz"),
    )
    parser.add_argument(
        "--out_file", type=pathlib.Path, default=pathlib.Path("hairs.npz")
    )
    args = parser.parse_args()

    results = calculate(args.parsing_results.resolve(strict=True))
    np.savez(args.out_file.resolve(), **results)
