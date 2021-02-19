import pathlib
import numpy as np
import tensorflow as tf
from typing import List


class ImagesGenerator:
    def __init__(self, model_path: pathlib.Path, seed: int, num_images: int) -> None:
        model = tf.saved_model.load(str(model_path))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(num_images, 512)
        self.dlatents = model.signatures["mapping"](
            latents=tf.convert_to_tensor(z, tf.float32)
        )["dlatents"].numpy()
        self.synthesis = model.signatures["synthesis"]

    def generate(self, offset: np.ndarray) -> List[np.ndarray]:
        results = []
        for dlatents_in in self.dlatents:
            images = self.synthesis(
                dlatents=tf.convert_to_tensor([dlatents_in + offset], tf.float32)
            )
            results.append(images["images"].numpy()[0])
        return results
