import argparse
import numpy as np
import tensorflow as tf


class LatentSpace(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(input_shape=())
        self.v = self.add_weight(
            shape=(1, 14, 512),
            dtype=tf.float32)

    def call(self, inputs):
        return tf.identity(self.v)


class Synthesis(tf.keras.layers.Layer):
    def __init__(self, model_path):
        super().__init__()
        model = tf.saved_model.load(model_path)
        self.synthesis = model.signatures['synthesis']

    def call(self, inputs):
        return self.synthesis(dlatents=inputs)['outputs']


class EmbeddingLoss(tf.keras.losses.Loss):
    def __init__(self, image):
        super().__init__()
        self.vgg16 = tf.keras.applications.VGG16(include_top=False)
        self.target_layers = {'block1_conv1', 'block1_conv2', 'block3_conv2', 'block4_conv2'}
        self.outputs = []
        out = image
        for layer in self.vgg16.layers:
            out = layer(out)
            if layer.name in self.target_layers:
                self.outputs.append(out)

    def call(self, y_true, y_pred):
        outputs = []
        out = y_pred
        for layer in self.vgg16.layers:
            out = layer(out)
            if layer.name in self.target_layers:
                outputs.append(out)
        n = tf.cast(tf.math.reduce_prod(y_pred.shape), tf.float32)
        losses = tf.math.reduce_sum(tf.math.squared_difference(y_true, y_pred)) / n
        for i, out in enumerate(outputs):
            n = tf.cast(tf.math.reduce_prod(out.shape), tf.float32)
            losses += tf.math.reduce_sum(tf.math.squared_difference(self.outputs[i], out)) / n
        return losses


class GenerateCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        v = self.model.layers[0].variables[0].numpy()
        images = self.model.layers[1](v)
        images = tf.saturate_cast((images + 1.0) * 127.5, tf.uint8)
        with open(f'epoch{epoch:03d}.png', 'wb') as fp:
            data = tf.image.encode_png(tf.squeeze(images, axis=0)).numpy()
            fp.write(data)


def run(model_path, target_image):
    tf.keras.backend.clear_session()

    model = tf.keras.Sequential([
        LatentSpace(),
        Synthesis(model_path),
    ])
    model.summary()

    with open(target_image, 'rb') as fp:
        y = tf.image.decode_jpeg(fp.read())
    y = tf.expand_dims(tf.cast(y, tf.float32) / 127.5 - 1.0, axis=0)

    dataset = tf.data.Dataset.from_tensors((0, y))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.01,
            epsilon=1e-08),
        loss=EmbeddingLoss(y))
    model.fit(
        dataset.repeat().batch(1),
        steps_per_epoch=50,
        epochs=100,
        callbacks=[GenerateCallback()])
    np.save('latents_in.npy', model.layers[0].variables[0].numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('target_image', type=str)
    args = parser.parse_args()

    run(args.model_path, args.target_image)
