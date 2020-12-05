import argparse
import os
import numpy as np
import tensorflow as tf
from datetime import datetime


def run(model_path, npz, outdir, seed, scale):
    vectors = np.load(npz)
    model = tf.saved_model.load(model_path)
    rnd = np.random.RandomState(seed)
    z = rnd.randn(1, 512)

    # calculate dlatents with mapping network
    mapping = model.signatures["mapping"]
    dlatents = mapping(latents=tf.convert_to_tensor(z, tf.float32))
    dlatents = np.squeeze(dlatents["dlatents"].numpy(), axis=0)

    # generate images
    synthesis = model.signatures["synthesis"]
    for emotion, vector in vectors.items():
        out_path = os.path.join(outdir, f"emotion_{emotion}_{scale:.02f}.png")
        now = datetime.now()
        images = synthesis(
            dlatents=tf.convert_to_tensor([dlatents + vector * scale], tf.float32)
        )
        images = images["images"].numpy()
        with open(out_path, "wb") as fp:
            fp.write(tf.io.encode_png(images[0]).numpy())
        elapsed = datetime.now() - now
        print(f"{out_path} ({elapsed})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("npz", type=str)
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--outdir", type=str, default="out")
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    run(args.model_path, args.npz, args.outdir, args.seed, args.scale)
