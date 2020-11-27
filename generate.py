import argparse
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from PIL import Image


def run(model_path, num_images, outdir, seed):
    model = tf.saved_model.load(model_path)
    generate = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    rnd = np.random.RandomState(seed)
    zs = rnd.randn(num_images, 1, 512)

    for i, z in enumerate(zs):
        out_path = os.path.join(outdir, f"generate_{i:05d}.png")
        now = datetime.now()
        images = generate(latents=tf.constant(z, tf.float32))["images"]
        Image.fromarray(images.numpy()[0]).save(out_path)
        elapsed = datetime.now() - now
        print(f"{out_path} ({elapsed})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--outdir", type=str, default="out")
    args = parser.parse_args()

    run(args.model_path, args.num, args.outdir, args.seed)
