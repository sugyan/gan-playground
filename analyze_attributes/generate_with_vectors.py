import argparse
import pathlib
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import List


class ImagesGenerator:
    def __init__(self, model_path: pathlib.Path, num_images: int, seed: int) -> None:
        rnd = np.random.RandomState(seed)
        z = rnd.randn(num_images, 512)

        model = tf.saved_model.load(str(model_path))
        # calculate dlatents with mapping network
        self.dlatents = model.signatures["mapping"](
            latents=tf.convert_to_tensor(z, tf.float32)
        )["dlatents"].numpy()
        self.synthesis = model.signatures["synthesis"]

    def __call__(self, offset: np.ndarray) -> List[np.ndarray]:
        return self.__generate(offset)

    def __generate(self, offset: np.ndarray) -> List[np.ndarray]:
        results = []
        for i, dlatents_in in enumerate(self.dlatents):
            now = datetime.now()
            images = self.synthesis(
                dlatents=tf.convert_to_tensor([dlatents_in + offset], tf.float32)
            )
            results.append(images["images"].numpy()[0])
            print(f"{i:05d}: {datetime.now() - now}")
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedmodel", type=pathlib.Path, required=True)
    parser.add_argument("--npz", type=pathlib.Path, required=True)
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with np.load(args.npz.resolve(strict=True)) as data:
        generator = ImagesGenerator(
            args.savedmodel.resolve(strict=True), args.num, args.seed
        )
        images = generator(data["emotion_happy"])
        tf.io.write_file("out.jpg", tf.io.encode_png(tf.concat(images, axis=1)))
