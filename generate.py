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
    save_dlatents: bool,
) -> None:
    model = tf.saved_model.load(str(model_path))
    rnd = np.random.RandomState(seed)
    z = rnd.randn(num_images, 512)

    # calculate dlatents with mapping network
    mapping = model.signatures["mapping"]
    dlatents = mapping(latents=tf.convert_to_tensor(z, tf.float32))
    dlatents = dlatents["dlatents"].numpy()

    # generate images
    synthesis = model.signatures["synthesis"]
    for i, dlatents_in in enumerate(dlatents):
        out_path = outdir.joinpath(f"generate_{i:05d}.png")

        now = datetime.now()
        images = synthesis(dlatents=tf.convert_to_tensor([dlatents_in], tf.float32))
        images = images["images"].numpy()
        Image.fromarray(images[0]).save(out_path)
        print(f"{out_path} ({datetime.now() - now})")
        if save_dlatents:
            np.save(f"{out_path}.npy", dlatents_in)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedmodel", type=pathlib.Path, required=True)
    parser.add_argument("--outdir", type=pathlib.Path, required=True)
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save_dlatents", action="store_true")
    args = parser.parse_args()

    run(
        args.savedmodel.resolve(strict=True),
        args.outdir.resolve(strict=True),
        args.num,
        args.seed,
        args.save_dlatents,
    )
