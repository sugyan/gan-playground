import argparse
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


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


def calculate_vectors(df: pd.DataFrame, rate: float) -> Dict[str, np.ndarray]:
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

        results[col] = np.array([np.mean(mins, axis=0), np.mean(maxs, axis=0)])

    return results


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

    results = calculate_vectors(df, args.rate)
    np.savez(
        args.out_file.resolve(),
        **results,
        mean=np.mean([np.load(f"{index}.npy") for index in df.index], axis=0),
    )
