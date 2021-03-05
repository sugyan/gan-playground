import pathlib
import numpy as np
import tensorflow as tf
from typing import Generator, List, Optional, Tuple


class ImagesGenerator:
    def __init__(self, model_path: pathlib.Path, seed: int, num_images: int) -> None:
        tf.compat.v1.enable_eager_execution()
        model = tf.compat.v2.saved_model.load(str(model_path))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(num_images, 512)
        self.dlatents = model.signatures["mapping"](
            latents=tf.convert_to_tensor(z, tf.float32)
        )["dlatents"].numpy()
        self.synthesis = model.signatures["synthesis"]

    def generate(
        self, offset: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        for dlatents_in in self.dlatents:
            inputs = np.array(dlatents_in)
            if offset is not None:
                inputs += offset
            images = self.synthesis(dlatents=tf.convert_to_tensor([inputs], tf.float32))
            yield images["images"].numpy()[0], inputs

    def images(self, offset: Optional[np.ndarray] = None) -> List[np.ndarray]:
        return [image for image, _ in self.generate(offset)]
