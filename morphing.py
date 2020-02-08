import argparse
import os
import numpy as np
import tensorflow as tf
from PIL import Image


def run(model_path, outdir):
    model = tf.saved_model.load(model_path)
    generate = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    rnd = np.random.RandomState(0)
    z = rnd.randn(3, 512)

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

    for i, latents_in in enumerate(inputs):
        out_path = os.path.join(outdir, f'{i:03d}.png')
        images = generate(latents=tf.constant([latents_in], tf.float32))['images']
        img = Image.fromarray(images.numpy()[0])
        img.save(out_path)
        print(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('--outdir', type=str, default='out')
    args = parser.parse_args()

    run(args.model_path, args.outdir)
