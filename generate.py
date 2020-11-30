import argparse
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from PIL import Image


def run(model_path, num_images, outdir, seed, save_dlatents):
    model = tf.saved_model.load(model_path)
    rnd = np.random.RandomState(seed)
    z = rnd.randn(num_images, 512)

    # calculate dlatents with mapping network
    mapping = model.signatures["mapping"]
    dlatents = mapping(latents=tf.convert_to_tensor(z, tf.float32))
    dlatents = dlatents["dlatents"].numpy()

    # generate images
    synthesis = model.signatures["synthesis"]
    for i, dlatents_in in enumerate(dlatents):
        out_path = os.path.join(outdir, f"generate_{i:05d}.png")
        now = datetime.now()
        images = synthesis(dlatents=tf.convert_to_tensor([dlatents_in], tf.float32))
        images = images["images"].numpy()
        Image.fromarray(images[0]).save(out_path)
        elapsed = datetime.now() - now
        print(f"{out_path} ({elapsed})")
        if save_dlatents:
            np.save(f"{out_path}.npy", dlatents_in)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--outdir", type=str, default="out")
    parser.add_argument("--save_dlatents", action="store_true")
    args = parser.parse_args()

    run(args.model_path, args.num, args.outdir, args.seed, args.save_dlatents)
