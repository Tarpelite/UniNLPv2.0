import shutil
from pathlib import Path
import onnx
import tensorflow as tf
from tensorflow.python.keras import backend as K
from onnx2keras import onnx_to_keras
import argparse


def pytorch2savedmodel(onnx_model_path, saved_model_dir):
    onnx_model = onnx.load(onnx_model_path)

    input_names = ['image_array']
    k_model = onnx_to_keras(onnx_model=onnx_model, input_names=input_names,
                            change_ordering=True, verbose=False)

    weights = k_model.get_weights()

    K.set_learning_phase(0)

    saved_model_dir = Path(saved_model_dir)
    if saved_model_dir.exists():
        shutil.rmtree(str(saved_model_dir))
    saved_model_dir.mkdir()

    with K.get_session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        k_model.set_weights(weights)

        tf.saved_model.simple_save(
            sess,
            str(saved_model_dir.joinpath('1')),
            inputs={'image_array': k_model.input},
            outputs=dict((output.name, tensor) for output, tensor in zip(onnx_model.graph.output, k_model.outputs))
        )


def savedmodel2tflite(saved_model_dir, tflite_model_path, quantize=False):
    saved_model_dir = str(Path(saved_model_dir).joinpath('1'))
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_model = converter.convert()

    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    return tflite_model

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--onnx_model_path")
    parser.add_argument("--saved_model_dir")
    parser.add_argument("--tflite_model_path")

    args = parser.parse_args()

    pytorch2savedmodel(args.onnx_model_path, args.saved_model_dir)
    savedmodel2tflite(args.saved_model_dir, args.tflite_model_path)

