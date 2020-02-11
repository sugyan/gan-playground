import argparse
import os
import pickle
import tempfile
import tensorflow as tf
from dnnlib import tflib, EasyDict


def convert(network_pkl, save_dir):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'generator')

        with tf.Graph().as_default() as graph:
            with tf.compat.v1.Session(graph=graph).as_default() as sess:
                with open(network_pkl, 'rb') as fp:
                    _, _, Gs = pickle.load(fp, encoding='latin1')
                # save weights
                saver = tf.compat.v1.train.Saver(Gs.vars)
                saver.save(sess, save_path)

        with tf.Graph().as_default() as graph:
            with tf.compat.v1.Session(graph=graph).as_default() as sess:
                # rebuild networks with custom options
                G_args = EasyDict(func_name='training.networks_stylegan2.G_main')
                G_args.fused_modconv = False
                G_args.randomize_noise = False
                G_args.return_dlatents = True
                G = tflib.Network(
                    'G',
                    num_channels=3,
                    resolution=256,
                    label_size=0,
                    **G_args)
                Gs = G.clone('Gs')
                # load weights
                saver = tf.compat.v1.train.Saver(Gs.vars)
                saver.restore(sess, save_path)
                # convert variables to constants
                input_names = [t.name for t in Gs.input_templates]
                output_names = [t.name for t in Gs.output_templates]
                graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
                    sess,
                    graph.as_graph_def(),
                    [t.op.name for t in Gs.output_templates])

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

        inputs = [graph.get_tensor_by_name(input_names[0])]
        outputs = [graph.get_tensor_by_name(name) for name in output_names]
        images = tf.transpose(outputs[0], [0, 2, 3, 1])
        images = tf.saturate_cast((images + 1.0) * 127.5, tf.uint8)
        # save as SavedModel
        builder = tf.compat.v1.saved_model.Builder(save_dir)
        default = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        signature_def_map = {
            default: tf.compat.v1.saved_model.build_signature_def(
                {'latents': tf.saved_model.utils.build_tensor_info(inputs[0])},
                {'images': tf.saved_model.utils.build_tensor_info(images)}),
            'mapping': tf.compat.v1.saved_model.build_signature_def(
                {'latents': tf.saved_model.utils.build_tensor_info(inputs[0])},
                {'dlatents': tf.saved_model.utils.build_tensor_info(outputs[1])}),
            'synthesis': tf.compat.v1.saved_model.build_signature_def(
                {'dlatents': tf.saved_model.utils.build_tensor_info(outputs[1])},
                {'images': tf.saved_model.utils.build_tensor_info(images),
                 'outputs': tf.saved_model.utils.build_tensor_info(tf.transpose(outputs[0], [0, 2, 3, 1]))})
        }
        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map)
        builder.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('network_pkl', type=str)
    parser.add_argument('save_dir', type=str)
    args = parser.parse_args()
    if os.path.exists(args.save_dir):
        print(f'save_dir {args.save_dir} already exists')
    else:
        convert(args.network_pkl, args.save_dir)
