import argparse
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from PIL import Image


def run(model_path, outdir):
    model = tf.saved_model.load(model_path)
    mapping = model.signatures['mapping']
    rnd = np.random.RandomState(0)
    z = rnd.randn(3, 512)
    z = mapping(latents=tf.constant(z, tf.float32))['dlatents'].numpy()

    step = 30
    inputs = []
    for i in range(step):
        rate = i / step
        inputs.append(z[0] + (z[1] - z[0]) * rate)
    for i in range(step):
        rate = i / step
        inputs.append(z[1] + (z[2] - z[1]) * rate)
    for i in range(step):
        rate = i / step
        inputs.append(z[2] + (z[0] - z[2]) * rate)

    synthesis = model.signatures['synthesis']
    for i, latents_in in enumerate(inputs):
        now = datetime.now()
        out_path = os.path.join(outdir, f'{i:03d}.png')
        images = synthesis(dlatents=tf.constant([latents_in], tf.float32))['images']
        img = Image.fromarray(images.numpy()[0])
        img.save(out_path)
        elapsed = datetime.now() - now
        print(f'{out_path} ({elapsed})')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('--outdir', type=str, default='out')
    args = parser.parse_args()

    run(args.model_path, args.outdir)
