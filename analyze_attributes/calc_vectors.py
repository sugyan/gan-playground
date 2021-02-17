import argparse
import pathlib
import numpy as np
import pandas as pd
from typing import List, Tuple


def _load_npz(filepath: pathlib.Path) -> Tuple[List[str], List[np.ndarray]]:
    d, index = [], []
    with np.load(filepath) as data:
        for k, v in data.items():
            d.append(v)
            index.append(k)
    return d, index


def load_emotion(filepath: pathlib.Path) -> pd.DataFrame:
    data, index = _load_npz(filepath)
    columns = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    return pd.DataFrame(data, index=index, columns=columns)


def load_hair(filepath: pathlib.Path) -> pd.DataFrame:
    data, index = _load_npz(filepath)
    columns = ["hair_volume", "hair_length", "hair_brightness"]
    return pd.DataFrame(data, index=index, columns=columns)


def load_headpose(filepath: pathlib.Path) -> pd.DataFrame:
    data, index = _load_npz(filepath)
    columns = ["yaw", "pitch", "roll"]
    return pd.DataFrame(data, index=index, columns=columns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emotion", type=pathlib.Path, required=False)
    parser.add_argument("--hair", type=pathlib.Path, required=False)
    parser.add_argument("--headpose", type=pathlib.Path, required=False)
    parser.add_argument(
        "--out_file", type=pathlib.Path, default=pathlib.Path("vectors.npz")
    )
    args = parser.parse_args()

    df = pd.DataFrame()
    if args.emotion:
        df = df.join(load_emotion(args.emotion.resolve(strict=True)), how="outer")
    if args.hair:
        df = df.join(load_hair(args.hair.resolve(strict=True)), how="outer")
    if args.headpose:
        df = df.join(load_headpose(args.headpose.resolve(strict=True)), how="outer")

    # TODO: all columns
    # all_dlatents = []
    # for index in df.index:
    #     all_dlatents.append(np.load(f"{index}.npy"))
    k = round(len(df) / 100.0)
    results = {}
    for col in ["hair_mask", "hair_length", "hair_brightness"]:
        print(f"top {k} images of {col}")
        print("min:")
        mins = []
        for index, row in df.sort_values(col, ascending=True)[:k].iterrows():
            print(index, row[col])
            mins.append(np.load(f"{index}.npy"))
        print("max:")
        maxs = []
        for index, row in df.sort_values(col, ascending=False)[:k].iterrows():
            print(index, row[col])
            maxs.append(np.load(f"{index}.npy"))
        results[col] = (np.mean(maxs, axis=0) - np.mean(mins, axis=0)) / 2.0

    np.savez(args.out_file.resolve(), **results)
