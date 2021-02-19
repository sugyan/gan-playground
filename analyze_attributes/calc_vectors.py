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
    return pd.DataFrame(data, index=index, columns=[f"emotion_{s}" for s in columns])


def load_hair(filepath: pathlib.Path) -> pd.DataFrame:
    data, index = _load_npz(filepath)
    columns = ["volume", "length", "brightness"]
    return pd.DataFrame(data, index=index, columns=[f"hair_{s}" for s in columns])


def load_headpose(filepath: pathlib.Path) -> pd.DataFrame:
    data, index = _load_npz(filepath)
    columns = ["yaw", "pitch", "roll"]
    return pd.DataFrame(data, index=index, columns=[f"headpose_{s}" for s in columns])


def calculate_vectors(df: pd.DataFrame, rate: float, out: pathlib.Path) -> None:
    k = round(len(df) * rate)
    results = {}
    for col in df.columns:
        print(f"Top {k} images of {col}")
        print("- MIN")
        mins = []
        for index, row in df.sort_values(col, ascending=True)[:k].iterrows():
            print(f"{index}: {row[col]}")
            mins.append(np.load(f"{index}.npy"))
        print("- MAX")
        maxs = []
        for index, row in df.sort_values(col, ascending=False)[:k].iterrows():
            print(f"{index}: {row[col]}")
            maxs.append(np.load(f"{index}.npy"))
        v = np.mean(maxs, axis=0) - np.mean(mins, axis=0)
        if not col.startswith("emotion"):
            v /= 2.0
        results[col] = v

    np.savez(args.out_file.resolve(), **results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emotion", type=pathlib.Path, required=False)
    parser.add_argument("--hair", type=pathlib.Path, required=False)
    parser.add_argument("--headpose", type=pathlib.Path, required=False)
    parser.add_argument("--rate", type=float, default=0.005)
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

    calculate_vectors(df, args.rate, args.out_file.resolve())
