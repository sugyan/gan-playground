import argparse
import pathlib
from typing import Optional

import numpy as np
from PIL import Image

from generator import ImagesGenerator


def main(
    savedmodel: pathlib.Path,
    seed: Optional[int],
    num: int,
    outdir: pathlib.Path,
    save_dlatents: bool,
) -> None:
    generator = ImagesGenerator(savedmodel, seed, num)
    for i, (image, dlatents) in enumerate(generator.generate()):
        out_path = outdir / f"generate_{i:05d}.png"
        print(out_path)
        Image.fromarray(image).save(out_path)
        if save_dlatents:
            np.save(f"{out_path}.npy", dlatents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedmodel", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--outdir", type=pathlib.Path, required=True)
    parser.add_argument("--save_dlatents", action="store_true")
    args = parser.parse_args()

    main(
        savedmodel=args.savedmodel.resolve(strict=True),
        seed=args.seed,
        num=args.num,
        outdir=args.outdir.resolve(strict=True),
        save_dlatents=args.save_dlatents,
    )
