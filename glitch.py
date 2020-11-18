import argparse
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from PIL import Image


def run(model_path, num_images, outdir, seed):
    model = tf.saved_model.load(model_path)
    rnd = np.random.RandomState(seed)

    # Probabilistically burst the mapping outputs
    mapping = model.signatures["mapping"]
    z = rnd.randn(1, 512)
    z = mapping(latents=tf.constant(z, tf.float32))["dlatents"].numpy()[0]
    inputs = []
    for i in range(num_images):
        inputs.append(z * np.where(rnd.uniform(size=z.shape) < 0.0001, 100.0, 1.0))

    # generate images
    synthesis = model.signatures["synthesis"]
    for i, latents_in in enumerate(inputs):
        now = datetime.now()
        out_path = os.path.join(outdir, f"glitch_{i:03d}.png")
        images = synthesis(dlatents=tf.constant([latents_in], tf.float32))["images"]
        img = Image.fromarray(images.numpy()[0])
        img.save(out_path)
        elapsed = datetime.now() - now
        print(f"{out_path} ({elapsed})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("--num", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--outdir", type=str, default="out")
    args = parser.parse_args()

    run(args.model_path, args.num, args.outdir, args.seed)
