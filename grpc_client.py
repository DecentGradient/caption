import grpc
import requests
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

# The image URL is the location of the image we should send to the server
IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'

# tf.app.flags.DEFINE_string('server', 'localhost:8500',
#                            'PredictionService host:port')
# tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS


def main(_):

  # Download the image since we weren't given one
  dl_request = requests.get(IMAGE_URL, stream=True)
  dl_request.raise_for_status()
  data = dl_request.content

  channel = grpc. insecure_channel("localhost:8500")    #insecure_channel("localhost:8500")
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  # Send request
  # See prediction_service.proto for gRPC request/response details.
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'caption'
  request.model_spec.signature_name = 'serving_default'
  request.inputs['image'].CopyFrom(
      tf.contrib.util.make_tensor_proto(data, shape=[1]))
  result = stub.Predict(request, 10.0)  # 10 secs timeout
  print(result)


if __name__ == '__main__':
  tf.app.run()