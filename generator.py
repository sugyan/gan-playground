import pathlib
from abc import ABC, abstractmethod
from typing import Generator, List, Optional, Tuple

import numpy as np
import tensorflow as tf


class StyleGAN(ABC):
    def __init__(self, model_path: pathlib.Path) -> None:
        self.model = tf.compat.v2.saved_model.load(str(model_path))

    @abstractmethod
    def mapping(self, latents: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def synthesis(self, dlatents: np.ndarray) -> np.ndarray:
        pass


class TF1Generator(StyleGAN):
    def __init__(self, model_path: pathlib.Path) -> None:
        self.sess = tf.compat.v1.Session(graph=tf.Graph())
        with self.sess.graph.as_default():
            super().__init__(model_path)

            signatures = self.model.signatures

            self.latents_in = tf.compat.v1.placeholder(tf.float32)
            self.dlatents = signatures["mapping"](self.latents_in)["dlatents"]

            self.dlatents_in = tf.compat.v1.placeholder(tf.float32)
            self.images = signatures["synthesis"](self.dlatents_in)["images"]

    def mapping(self, latents: np.ndarray) -> np.ndarray:
        return self.sess.run(self.dlatents, feed_dict={self.latents_in: latents})

    def synthesis(self, dlatents: np.ndarray) -> np.ndarray:
        return self.sess.run(self.images, feed_dict={self.dlatents_in: [dlatents]})

    def __del__(self) -> None:
        self.sess.close()


class TF2Generator(StyleGAN):
    def mapping(self, latents: np.ndarray) -> np.ndarray:
        latents = tf.convert_to_tensor(latents, dtype=tf.float32)
        return self.model.signatures["mapping"](latents)["dlatents"].numpy()

    def synthesis(self, dlatents: np.ndarray) -> np.ndarray:
        dlatents = tf.convert_to_tensor([dlatents], dtype=tf.float32)
        return self.model.signatures["synthesis"](dlatents)["images"].numpy()


class ImagesGenerator:
    def __init__(
        self, model_path: pathlib.Path, seed: Optional[int], num_images: int
    ) -> None:
        self.g: StyleGAN
        if tf.__version__.startswith("2."):
            self.g = TF2Generator(model_path)
        else:
            self.g = TF1Generator(model_path)
        rnd = np.random.RandomState(seed)
        self.z = rnd.randn(num_images, 512)
        self.dlatents = self.g.mapping(latents=self.z)

    def generate(
        self, offset: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        for dlatents_in in self.dlatents:
            inputs = np.array(dlatents_in)
            if offset is not None:
                inputs += offset
            images = self.g.synthesis(inputs)
            yield images[0], inputs

    def images(self, offset: Optional[np.ndarray] = None) -> List[np.ndarray]:
        return [image for image, _ in self.generate(offset)]
