import argparse
import pathlib
import numpy as np
import tensorflow as tf
from generator import ImagesGenerator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedmodel", type=pathlib.Path, required=True)
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--outdir", type=pathlib.Path, required=True)
    parser.add_argument("--save_dlatents", action="store_true")
    args = parser.parse_args()

    generator = ImagesGenerator(
        args.savedmodel.resolve(strict=True), args.seed, args.num
    )
    for i, (image, dlatents) in enumerate(generator.generate()):
        out_path = str(args.outdir.resolve(strict=True) / f"generate_{i:05d}.png")
        print(out_path)
        tf.io.write_file(out_path, tf.io.encode_png(image))
        if args.save_dlatents:
            np.save(f"{out_path}.npy", dlatents)
