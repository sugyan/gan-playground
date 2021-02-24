import argparse
import pathlib
import numpy as np
import tensorflow as tf

from generator import ImagesGenerator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedmodel", type=pathlib.Path, required=True)
    parser.add_argument("--npz", type=pathlib.Path, required=True)
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--outdir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    generator = ImagesGenerator(
        args.savedmodel.resolve(strict=True), args.seed, args.num
    )
    with np.load(args.npz.resolve(strict=True)) as data:
        for scale in [x / 10.0 for x in range(11)]:
            print(scale)
            vectors = [
                data["emotion_happy"],
                data["headpose_pitch"] - data["headpose_yaw"],
                data["hair_length"] + data["hair_brightness"],
            ]
            tile = tf.concat(
                [tf.concat(generator.images(v * scale), axis=1) for v in vectors],
                axis=0,
            )
            tf.io.write_file(
                str(args.outdir.resolve(strict=True) / f"tile_{scale:.2f}.png"),
                tf.io.encode_png(tile),
            )
