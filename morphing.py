import argparse
import pathlib
import numpy as np
import tensorflow as tf
from datetime import datetime
from PIL import Image


def run(
    model_path: pathlib.Path,
    outdir: pathlib.Path,
    num_images: int,
    seed: int,
    steps: int,
) -> None:
    model = tf.saved_model.load(str(model_path))
    rnd = np.random.RandomState(seed)

    # generate dlatents from mapping outputs
    mapping = model.signatures["mapping"]
    z = rnd.randn(num_images, 512)
    z = mapping(latents=tf.constant(z, tf.float32))["dlatents"].numpy()
    inputs = []
    for i in range(num_images):
        j = (i + 1) % num_images
        for s in range(steps):
            t = s / steps
            inputs.append((1 - t) * z[i] + t * z[j])

    # generate images
    synthesis = model.signatures["synthesis"]
    for i, latents_in in enumerate(inputs):
        out_path = outdir / f"morphing_{i:03d}.png"

        now = datetime.now()
        images = synthesis(dlatents=tf.constant([latents_in], tf.float32))["images"]
        Image.fromarray(images.numpy()[0]).save(out_path)
        print(f"{out_path} ({datetime.now() - now})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedmodel", type=pathlib.Path, required=True)
    parser.add_argument("--outdir", type=pathlib.Path, required=True)
    parser.add_argument("--num", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    run(
        args.savedmodel.resolve(strict=True),
        args.outdir.resolve(strict=True),
        args.num,
        args.seed,
        args.steps,
    )
