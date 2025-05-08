from onnx_tf.backend import prepare
import onnx

onnx_model_path = 'yolov8_model.onnx'
tf_model_path = 'yolov8_model_tf'

onnx_model = onnx.load(onnx_model_path)
tf_rep = prepare(onnx_model)
tf_rep.export_graph(tf_model_path)