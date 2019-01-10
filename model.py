  # Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os


import tensorflow as tf
from tensorflow.python.saved_model import simple_save

from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base

def parse_sequence_example(serialized, image_feature, caption_feature):
  """Parses a tensorflow.SequenceExample into an image and caption.

  Args:
    serialized: A scalar string Tensor; a single serialized SequenceExample.
    image_feature: Name of SequenceExample context feature containing image
      data.
    caption_feature: Name of SequenceExample feature list containing integer
      captions.

  Returns:
    encoded_image: A scalar string Tensor containing a JPEG encoded image.
    caption: A 1-D uint64 Tensor with dynamically specified length.
  """
  context, sequence = tf.parse_single_sequence_example(
      serialized,
      context_features={
          image_feature: tf.FixedLenFeature([], dtype=tf.string)
      },
      sequence_features={
          caption_feature: tf.FixedLenSequenceFeature([], dtype=tf.int64),
      })

  encoded_image = context[image_feature]
  caption = sequence[caption_feature]
  return encoded_image, caption


def prefetch_input_data(reader,
                        file_pattern,
                        is_training,
                        batch_size,
                        values_per_shard,
                        input_queue_capacity_factor=16,
                        num_reader_threads=1,
                        shard_queue_name="filename_queue",
                        value_queue_name="input_queue"):
  """Prefetches string values from disk into an input queue.

  In training the capacity of the queue is important because a larger queue
  means better mixing of training examples between shards. The minimum number of
  values kept in the queue is values_per_shard * input_queue_capacity_factor,
  where input_queue_memory factor should be chosen to trade-off better mixing
  with memory usage.

  Args:
    reader: Instance of tf.ReaderBase.
    file_pattern: Comma-separated list of file patterns (e.g.
        /tmp/train_data-?????-of-00100).
    is_training: Boolean; whether prefetching for training or eval.
    batch_size: Model batch size used to determine queue capacity.
    values_per_shard: Approximate number of values per shard.
    input_queue_capacity_factor: Minimum number of values to keep in the queue
      in multiples of values_per_shard. See comments above.
    num_reader_threads: Number of reader threads to fill the queue.
    shard_queue_name: Name for the shards filename queue.
    value_queue_name: Name for the values input queue.

  Returns:
    A Queue containing prefetched string values.
  """
  data_files = []
  for pattern in file_pattern.split(","):
    data_files.extend(tf.gfile.Glob(pattern))
  if not data_files:
    tf.logging.fatal("Found no input files matching %s", file_pattern)
  else:
    tf.logging.info("Prefetching values from %d files matching %s",
                    len(data_files), file_pattern)

  if is_training:
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=True, capacity=16, name=shard_queue_name)
    min_queue_examples = values_per_shard * input_queue_capacity_factor
    capacity = min_queue_examples + 100 * batch_size
    values_queue = tf.RandomShuffleQueue(
        capacity=capacity,
        min_after_dequeue=min_queue_examples,
        dtypes=[tf.string],
        name="random_" + value_queue_name)
  else:
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=False, capacity=1, name=shard_queue_name)
    capacity = values_per_shard + 3 * batch_size
    values_queue = tf.FIFOQueue(
        capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name)

  enqueue_ops = []
  for _ in range(num_reader_threads):
    _, value = reader.read(filename_queue)
    enqueue_ops.append(values_queue.enqueue([value]))
  tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
      values_queue, enqueue_ops))
  tf.summary.scalar(
      "queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
      tf.cast(values_queue.size(), tf.float32) * (1. / capacity))

  return values_queue


def batch_with_dynamic_pad(images_and_captions,
                           batch_size,
                           queue_capacity,
                           add_summaries=True):
  """Batches input images and captions.

  This function splits the caption into an input sequence and a target sequence,
  where the target sequence is the input sequence right-shifted by 1. Input and
  target sequences are batched and padded up to the maximum length of sequences
  in the batch. A mask is created to distinguish real words from padding words.

  Example:
    Actual captions in the batch ('-' denotes padded character):
      [
        [ 1 2 3 4 5 ],
        [ 1 2 3 4 - ],
        [ 1 2 3 - - ],
      ]

    input_seqs:
      [
        [ 1 2 3 4 ],
        [ 1 2 3 - ],
        [ 1 2 - - ],
      ]

    target_seqs:
      [
        [ 2 3 4 5 ],
        [ 2 3 4 - ],
        [ 2 3 - - ],
      ]

    mask:
      [
        [ 1 1 1 1 ],
        [ 1 1 1 0 ],
        [ 1 1 0 0 ],
      ]

  Args:
    images_and_captions: A list of pairs [image, caption], where image is a
      Tensor of shape [height, width, channels] and caption is a 1-D Tensor of
      any length. Each pair will be processed and added to the queue in a
      separate thread.
    batch_size: Batch size.
    queue_capacity: Queue capacity.
    add_summaries: If true, add caption length summaries.

  Returns:
    images: A Tensor of shape [batch_size, height, width, channels].
    input_seqs: An int32 Tensor of shape [batch_size, padded_length].
    target_seqs: An int32 Tensor of shape [batch_size, padded_length].
    mask: An int32 0/1 Tensor of shape [batch_size, padded_length].
  """
  enqueue_list = []
  for image, caption in images_and_captions:
    caption_length = tf.shape(caption)[0]
    input_length = tf.expand_dims(tf.subtract(caption_length, 1), 0)

    input_seq = tf.slice(caption, [0], input_length)
    target_seq = tf.slice(caption, [1], input_length)
    indicator = tf.ones(input_length, dtype=tf.int32)
    enqueue_list.append([image, input_seq, target_seq, indicator])

  images, input_seqs, target_seqs, mask = tf.train.batch_join(
      enqueue_list,
      batch_size=batch_size,
      capacity=queue_capacity,
      dynamic_pad=True,
      name="batch_and_pad")

  if add_summaries:
    lengths = tf.add(tf.reduce_sum(mask, 1), 1)
    tf.summary.scalar("caption_length/batch_min", tf.reduce_min(lengths))
    tf.summary.scalar("caption_length/batch_max", tf.reduce_max(lengths))
    tf.summary.scalar("caption_length/batch_mean", tf.reduce_mean(lengths))

  return images, input_seqs, target_seqs, mask

class Vocabulary(object):
  """Vocabulary class for an image-to-text model."""

  def __init__(self,
               vocab_file,
               start_word="<S>",
               end_word="</S>",
               unk_word="<UNK>"):
    """Initializes the vocabulary.

    Args:
      vocab_file: File containing the vocabulary, where the words are the first
        whitespace-separated token on each line (other tokens are ignored) and
        the word ids are the corresponding line numbers.
      start_word: Special word denoting sentence start.
      end_word: Special word denoting sentence end.
      unk_word: Special word denoting unknown words.
    """
    if not tf.gfile.Exists(vocab_file):
      tf.logging.fatal("Vocab file %s not found.", vocab_file)
    tf.logging.info("Initializing vocabulary from file: %s", vocab_file)

    with tf.gfile.GFile(vocab_file, mode="r") as f:
      reverse_vocab = list(f.readlines())
    reverse_vocab = [line.split()[0] for line in reverse_vocab]
    assert start_word in reverse_vocab
    assert end_word in reverse_vocab
    if unk_word not in reverse_vocab:
      reverse_vocab.append(unk_word)
    vocab = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])

    tf.logging.info("Created vocabulary with %d words" % len(vocab))

    self.vocab = vocab  # vocab[word] = id
    self.reverse_vocab = reverse_vocab  # reverse_vocab[id] = word

    # Save special word ids.
    self.start_id = vocab[start_word]
    self.end_id = vocab[end_word]
    self.unk_id = vocab[unk_word]

  def word_to_id(self, word):
    """Returns the integer word id of a word string."""
    if word in self.vocab:
      return self.vocab[word]
    else:
      return self.unk_id

  def id_to_word(self, word_id):
    """Returns the word string of an integer word id."""
    if word_id >= len(self.reverse_vocab):
      return self.reverse_vocab[self.unk_id]
    else:
      return self.reverse_vocab[word_id]

slim = tf.contrib.slim
def process_image(encoded_image,
                  is_training,
                  height,
                  width,
                  resize_height=346,
                  resize_width=346,
                  thread_id=0,
                  image_format="jpeg"):
  """Decode an image, resize and apply random distortions.

  In training, images are distorted slightly differently depending on thread_id.

  Args:
    encoded_image: String Tensor containing the image.
    is_training: Boolean; whether preprocessing for training or eval.
    height: Height of the output image.
    width: Width of the output image.
    resize_height: If > 0, resize height before crop to final dimensions.
    resize_width: If > 0, resize width before crop to final dimensions.
    thread_id: Preprocessing thread id used to select the ordering of color
      distortions. There should be a multiple of 2 preprocessing threads.
    image_format: "jpeg" or "png".

  Returns:
    A float32 Tensor of shape [height, width, 3] with values in [-1, 1].

  Raises:
    ValueError: If image_format is invalid.
  """
  # Helper function to log an image summary to the visualizer. Summaries are
  # only logged in thread 0.
  def image_summary(name, image):
    if not thread_id:
      tf.summary.image(name, tf.expand_dims(image, 0))

  # Decode image into a float32 Tensor of shape [?, ?, 3] with values in [0, 1).
  with tf.name_scope("decode", values=[encoded_image]):
    if image_format == "jpeg":
      image = tf.image.decode_jpeg(encoded_image, channels=3)
    elif image_format == "png":
      image = tf.image.decode_png(encoded_image, channels=3)
    else:
      raise ValueError("Invalid image format: %s" % image_format)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  image_summary("original_image", image)

  # Resize image.
  assert (resize_height > 0) == (resize_width > 0)
  if resize_height:
    image = tf.image.resize_images(image,
                                   size=[resize_height, resize_width],
                                   method=tf.image.ResizeMethod.BILINEAR)

  # Crop to final dimensions.
  if is_training:
    image = tf.random_crop(image, [height, width, 3])
  else:
    # Central crop, assuming resize_height > height, resize_width > width.
    image = tf.image.resize_image_with_crop_or_pad(image, height, width)

  image_summary("resized_image", image)

  # Randomly distort the image.
  # if is_training:
  #   image = distort_image(image, thread_id)

  image_summary("final_image", image)

  # Rescale to [-1,1] instead of [0, 1]
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)
  return image


def inception_v3(images,
                 trainable=True,
                 is_training=True,
                 weight_decay=0.00004,
                 stddev=0.1,
                 dropout_keep_prob=0.8,
                 use_batch_norm=True,
                 batch_norm_params=None,
                 add_summaries=True,
                 scope="InceptionV3"):
  """Builds an Inception V3 subgraph for image embeddings.

  Args:
    images: A float32 Tensor of shape [batch, height, width, channels].
    trainable: Whether the inception submodel should be trainable or not.
    is_training: Boolean indicating training mode or not.
    weight_decay: Coefficient for weight regularization.
    stddev: The standard deviation of the trunctated normal weight initializer.
    dropout_keep_prob: Dropout keep probability.
    use_batch_norm: Whether to use batch normalization.
    batch_norm_params: Parameters for batch normalization. See
      tf.contrib.layers.batch_norm for details.
    add_summaries: Whether to add activation summaries.
    scope: Optional Variable scope.

  Returns:
    end_points: A dictionary of activations from inception_v3 layers.
  """
  # Only consider the inception model to be in training mode if it's trainable.
  is_inception_model_training = trainable and is_training

  if use_batch_norm:
    # Default parameters for batch normalization.
    if not batch_norm_params:
      batch_norm_params = {
          "is_training": is_inception_model_training,
          "trainable": trainable,
          # Decay for the moving averages.
          "decay": 0.9997,
          # Epsilon to prevent 0s in variance.
          "epsilon": 0.001,
          # Collection containing the moving mean and moving variance.
          "variables_collections": {
              "beta": None,
              "gamma": None,
              "moving_mean": ["moving_vars"],
              "moving_variance": ["moving_vars"],
          }
      }
  else:
    batch_norm_params = None

  if trainable:
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  else:
    weights_regularizer = None

  with tf.variable_scope(scope, "InceptionV3", [images]) as scope:
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer=weights_regularizer,
        trainable=trainable):
      with slim.arg_scope(
          [slim.conv2d],
          weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
          activation_fn=tf.nn.relu,
          normalizer_fn=slim.batch_norm,
          normalizer_params=batch_norm_params):
        net, end_points = inception_v3_base(images, scope=scope)
        with tf.variable_scope("logits"):
          shape = net.get_shape()
          net = slim.avg_pool2d(net, shape[1:3], padding="VALID", scope="pool")
          net = slim.dropout(
              net,
              keep_prob=dropout_keep_prob,
              is_training=is_inception_model_training,
              scope="dropout")
          net = slim.flatten(net, scope="flatten")

  # Add summaries.
  if add_summaries:
    for v in end_points.values():
      tf.contrib.layers.summaries.summarize_activation(v)

  return net


class InferenceWrapperBase(object):
  """Base wrapper class for performing inference with an image-to-text model."""

  def __init__(self):
    pass

  def build_model(self, model_config):
    """Builds the model for inference.

    Args:
      model_config: Object containing configuration for building the model.

    Returns:
      model: The model object.
    """
    tf.logging.fatal("Please implement build_model in subclass")


  def build_graph_from_proto(self, graph_def_file, saver_def_file,
                             checkpoint_path):
    """Builds the inference graph from serialized GraphDef and SaverDef protos.

    Args:
      graph_def_file: File containing a serialized GraphDef proto.
      saver_def_file: File containing a serialized SaverDef proto.
      checkpoint_path: Checkpoint file or a directory containing a checkpoint
        file.

    Returns:
      restore_fn: A function such that restore_fn(sess) loads model variables
        from the checkpoint file.
    """
    # Load the Graph.
    tf.logging.info("Loading GraphDef from file: %s", graph_def_file)
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(graph_def_file, "rb") as f:
      graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name="")

    # Load the Saver.
    tf.logging.info("Loading SaverDef from file: %s", saver_def_file)
    saver_def = tf.train.SaverDef()
    with tf.gfile.FastGFile(saver_def_file, "rb") as f:
      saver_def.ParseFromString(f.read())
    saver = tf.train.Saver(saver_def=saver_def)

    return self._create_restore_fn(checkpoint_path, saver)

  def feed_image(self, sess, encoded_image):
    """Feeds an image and returns the initial model state.

    See comments at the top of file.

    Args:
      sess: TensorFlow Session object.
      encoded_image: An encoded image string.

    Returns:
      state: A numpy array of shape [1, state_size].
    """
    tf.logging.fatal("Please implement feed_image in subclass")

  def inference_step(self, sess, input_feed, state_feed):
    """Runs one step of inference.

    Args:
      sess: TensorFlow Session object.
      input_feed: A numpy array of shape [batch_size].
      state_feed: A numpy array of shape [batch_size, state_size].

    Returns:
      softmax_output: A numpy array of shape [batch_size, vocab_size].
      new_state: A numpy array of shape [batch_size, state_size].
      metadata: Optional. If not None, a string containing metadata about the
        current inference step (e.g. serialized numpy array containing
        activations from a particular model layer.).
    """
    tf.logging.fatal("Please implement inference_step in subclass")

class ModelConfig(object):
  """Wrapper class for model hyperparameters."""

  def __init__(self):
    """Sets the default model hyperparameters."""
    # File pattern of sharded TFRecord file containing SequenceExample protos.
    # Must be provided in training and evaluation modes.
    self.input_file_pattern = None

    # Image format ("jpeg" or "png").
    self.image_format = "jpeg"

    # Approximate number of values per input shard. Used to ensure sufficient
    # mixing between shards in training.
    self.values_per_input_shard = 2300
    # Minimum number of shards to keep in the input queue.
    self.input_queue_capacity_factor = 2
    # Number of threads for prefetching SequenceExample protos.
    self.num_input_reader_threads = 1

    # Name of the SequenceExample context feature containing image data.
    self.image_feature_name = "image/data"
    # Name of the SequenceExample feature list containing integer captions.
    self.caption_feature_name = "image/caption_ids"

    # Number of unique words in the vocab (plus 1, for <UNK>).
    # The default value is larger than the expected actual vocab size to allow
    # for differences between tokenizer versions used in preprocessing. There is
    # no harm in using a value greater than the actual vocab size, but using a
    # value less than the actual vocab size will result in an error.
    self.vocab_size = 12000

    # Number of threads for image preprocessing. Should be a multiple of 2.
    self.num_preprocess_threads = 4

    # Batch size.
    self.batch_size = 32

    # File containing an Inception v3 checkpoint to initialize the variables
    # of the Inception model. Must be provided when starting training for the
    # first time.
    self.inception_checkpoint_file = None

    # Dimensions of Inception v3 input images.
    self.image_height = 299
    self.image_width = 299

    # Scale used to initialize model variables.
    self.initializer_scale = 0.08

    # LSTM input and output dimensionality, respectively.
    self.embedding_size = 512
    self.num_lstm_units = 512

    # If < 1.0, the dropout keep probability applied to LSTM variables.
    self.lstm_dropout_keep_prob = 0.7
class ShowAndTellModel(InferenceWrapperBase):
  """Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

  "Show and Tell: A Neural Image Caption Generator"
  Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
  """

  def __init__(self, config, mode, train_inception=False,checkpoint = None):
    """Basic setup.

    Args:
      config: Object containing configuration parameters.
      mode: "train", "eval" or "inference".
      train_inception: Whether the inception submodel variables are trainable.
    """
    super(ShowAndTellModel, self).__init__()
    assert mode in ["train", "eval", "inference"]
    self.config = config
    self.mode = mode
    self.train_inception = train_inception

    # Reader for the input data.
    self.reader = tf.TFRecordReader()

    # To match the "Show and Tell" paper we initialize all variables with a
    # random uniform initializer.
    self.initializer = tf.random_uniform_initializer(
        minval=-self.config.initializer_scale,
        maxval=self.config.initializer_scale)

    # A float32 Tensor with shape [batch_size, height, width, channels].
    self.images = None

    # An int32 Tensor with shape [batch_size, padded_length].
    self.input_seqs = None

    # An int32 Tensor with shape [batch_size, padded_length].
    self.target_seqs = None

    # An int32 0/1 Tensor with shape [batch_size, padded_length].
    self.input_mask = None

    # A float32 Tensor with shape [batch_size, embedding_size].
    self.image_embeddings = None

    # A float32 Tensor with shape [batch_size, padded_length, embedding_size].
    self.seq_embeddings = None

    # A float32 scalar Tensor; the total loss for the trainer to optimize.
    self.total_loss = None

    # A float32 Tensor with shape [batch_size * padded_length].
    self.target_cross_entropy_losses = None

    # A float32 Tensor with shape [batch_size * padded_length].
    self.target_cross_entropy_loss_weights = None

    # Collection of variables from the inception submodel.
    self.inception_variables = []

    # Function to restore the inception submodel from checkpoint.
    self.init_fn = None

    # Global step Tensor.
    self.global_step = None
    self.image_feed = tf.placeholder(dtype=tf.string, shape=(None), name="image_bytes")
  def _create_restore_fn(self, checkpoint_path, saver):
    """Creates a function that restores a model from checkpoint.

    Args:
      checkpoint_path: Checkpoint file or a directory containing a checkpoint
        file.
      saver: Saver for restoring variables from the checkpoint file.

    Returns:
      restore_fn: A function such that restore_fn(sess) loads model variables
        from the checkpoint file.

    Raises:
      ValueError: If checkpoint_path does not refer to a checkpoint file or a
        directory containing a checkpoint file.
    """
    if tf.gfile.IsDirectory(checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
      if not checkpoint_path:
        raise ValueError("No checkpoint file found in: %s" % checkpoint_path)

    def _restore_fn(sess):
      tf.logging.info("Loading model from checkpoint: %s", checkpoint_path)
      saver.restore(sess, checkpoint_path)
      tf.logging.info("Successfully loaded checkpoint: %s",
                      os.path.basename(checkpoint_path))

    return _restore_fn

  def build_graph_from_config(self, checkpoint_path):
    """Builds the inference graph from a configuration object.

    Args:
      model_config: Object containing configuration for building the model.
      checkpoint_path: Checkpoint file or a directory containing a checkpoint
        file.

    Returns:
      restore_fn: A function such that restore_fn(sess) loads model variables
        from the checkpoint file.
    """
    tf.logging.info("Building model.")
    self.build_self(self.config)
    from tensorflow.python import pywrap_tensorflow
    def print_tensors_in_checkpoint_file(file_name, tensor_name, all_tensors):
        varlist = []
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        if all_tensors:
            var_to_shape_map = reader.get_variable_to_shape_map()
            for key in sorted(var_to_shape_map):
                varlist.append(key)
        return varlist

    varlist = print_tensors_in_checkpoint_file(file_name=checkpoint_path, all_tensors = True, tensor_name = None)
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    globalstep = variables[-1]
    self.sentence = variables[-2]

    variables = variables[:len(varlist) - 1]
    variables.append(globalstep)
    saver = tf.train.Saver(variables)

    return self._create_restore_fn(checkpoint_path, saver)


  def is_training(self):
    """Returns true if the model is built for training mode."""
    return self.mode == "train"

  def process_image(self, encoded_image, thread_id=0):
    """Decodes and processes an image string.

    Args:
      encoded_image: A scalar string Tensor; the encoded image.
      thread_id: Preprocessing thread id used to select the ordering of color
        distortions.

    Returns:
      A float32 Tensor of shape [height, width, 3]; the processed image.
    """
    return process_image(encoded_image,
                                          is_training=self.is_training(),
                                          height=self.config.image_height,
                                          width=self.config.image_width,
                                          thread_id=thread_id,
                                          image_format=self.config.image_format)

  def build_inputs(self):
    """Input prefetching, preprocessing and batching.

    Outputs:
      self.images
      self.input_seqs
      self.target_seqs (training and eval only)
      self.input_mask (training and eval only)
    """
    # In inference mode, images and inputs are fed via placeholders.
    image_feed = tf.squeeze(self.image_feed)
    input_feed = [0]
    #input_feed = tf.placeholder(dtype=tf.int64,
    #                           shape=[None],  # batch_size
    #                          name="input_feed")
    self.input_feed = input_feed
    # Process image and insert batch dimensions.
    images = tf.expand_dims(self.process_image(image_feed), 0)
    input_seqs = tf.expand_dims(input_feed, 1)

    # No target sequences or input mask in inference mode.
    target_seqs = None
    input_mask = None

    self.images = images
    self.input_seqs = input_seqs
    self.target_seqs = target_seqs
    self.input_mask = input_mask

  def build_image_embeddings(self):
    """Builds the image model subgraph and generates image embeddings.

    Inputs:
      self.images

    Outputs:
      self.image_embeddings
    """
    inception_output = inception_v3(
        self.images,
        trainable=self.train_inception,
        is_training=self.is_training())
    self.inception_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")

    # Map inception output into embedding space.
    with tf.variable_scope("image_embedding") as scope:
      image_embeddings = tf.contrib.layers.fully_connected(
          inputs=inception_output,
          num_outputs=self.config.embedding_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          biases_initializer=None,
          scope=scope)

    # Save the embedding size in the graph.
    tf.constant(self.config.embedding_size, name="embedding_size")

    self.image_embeddings = image_embeddings

  def build_seq_embeddings(self):
    """Builds the input sequence embeddings.

    Inputs:
      self.input_seqs

    Outputs:
      self.seq_embeddings
    """
    with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
      embedding_map = tf.get_variable(
          name="map",
          shape=[self.config.vocab_size, self.config.embedding_size],
          initializer=self.initializer)
      seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.input_seqs)

    self.seq_embeddings = seq_embeddings
    self.embedding_map  = embedding_map


  def build_self(self, model_config):
    #model = ShowAndTellModel(model_config, mode="inference")
    out = self.build()
    self.out = out
    return out

  def feed_image(self, sess, encoded_image):
    initial_state = sess.run(fetches="lstm/initial_state:0",
                             feed_dict={"image_bytes:0": encoded_image})
    return initial_state

  def inference_step(self, sess, input_feed, state_feed):
    softmax_output, state_output = sess.run(
        fetches=["softmax:0", "lstm/state:0"],
        feed_dict={
            "input_feed:0": input_feed,
            "lstm/state_feed:0": state_feed,
        })
    return softmax_output, state_output, None
  #@autograph.convert(verbose=True)
  def build_model(self):
    """Builds the model.

    Inputs:
      self.image_embeddings
      self.seq_embeddings
      self.target_seqs (training and eval only)
      self.input_mask (training and eval only)

    Outputs:
      self.total_loss (training and eval only)
      self.target_cross_entropy_losses (training and eval only)
      self.target_cross_entropy_loss_weights (training and eval only)
    """
    # This LSTM cell has biases and outputs tanh(new_c) * sigmoid(o), but the
    # modified LSTM in the "Show and Tell" paper has no biases and outputs
    # new_c * sigmoid(o).
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(
        num_units=self.config.num_lstm_units, state_is_tuple=True)
    vocab = Vocabulary(FLAGS.vocab_file)

    with tf.variable_scope("lstm", initializer=self.initializer) as lstm_scope:
      # Feed the image embeddings to set the initial LSTM state.
      zero_state = lstm_cell.zero_state(
          batch_size=self.image_embeddings.get_shape()[0], dtype=tf.float32)
      _, initial_state = lstm_cell(self.image_embeddings, zero_state)

      # Allow the LSTM variables to be reused.
      lstm_scope.reuse_variables()
      input_seqs = tf.expand_dims([vocab.start_id], 1)

      seq_embeddings = tf.nn.embedding_lookup(self.embedding_map, input_seqs)
      embed = seq_embeddings
      state = initial_state
        # In inference mode, use concatenated states for convenient feeding and
        # fetching.
      state_feed = tf.concat(axis=1, values=state, name="initial_state")

        # Placeholder for feeding a batch of concatenated states.
        # state_feed = tf.placeholder(dtype=tf.float32,
        #                             shape=[None, sum(lstm_cell.state_size)],
        #                             name="state_feed")
      state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)
      state = state_tuple
        # Run a single LSTM step.
      lstm_outputs, state_tuple = lstm_cell(
            inputs=tf.squeeze(embed, axis=[1]),
            state=state_tuple)

        # Concatentate the resulting state.
      state = tf.concat(axis=1, values=state_tuple, name="state")

    # Stack batches vertically.
    lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

    with tf.variable_scope("logits") as logits_scope:
      logits = tf.contrib.layers.fully_connected(
          inputs=lstm_outputs,
          num_outputs=self.config.vocab_size,
          activation_fn=None,
          #weights_initializer=self.initializer,
          scope=logits_scope)


    softmax = tf.nn.softmax(logits, name="softmax")
    self.softmax = softmax
    val, idx =tf.nn.top_k(softmax ,name="topk")
    sentence = tf.Variable([vocab.start_id],False,name="sentence",)
    sentence = tf.concat([sentence, idx[0]], 0)#

    def cond(sentence,state):
        return tf.not_equal( sentence[-1],tf.constant(vocab.end_id))

    def body(sentence,state):
        input_seqs = tf.expand_dims([sentence[-1]], 1)

        seq_embeddings = tf.nn.embedding_lookup(self.embedding_map, input_seqs)
        embed = seq_embeddings

        # In inference mode, use concatenated states for convenient feeding and
        # fetching.
        state_feed = tf.concat(axis=1, values=state, name="state")

        # Placeholder for feeding a batch of concatenated states.
        # state_feed = tf.placeholder(dtype=tf.float32,
        #                             shape=[None, sum(lstm_cell.state_size)],
        #                             name="state_feed")
        state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)

        # Run a single LSTM step.
        lstm_outputs, new_state_tuple = lstm_cell(
            inputs=tf.squeeze(embed, axis=[1]),
            state=state_tuple)

        # Concatentate the resulting state.
        state = tf.concat(axis=1, values=new_state_tuple, name="state")

        # Stack batches vertically.

        lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

        with tf.variable_scope("logits") as logits_scope:
            logits = tf.contrib.layers.fully_connected(
            inputs=lstm_outputs,
            num_outputs=self.config.vocab_size,
            activation_fn=None,
            weights_initializer=self.initializer,
            scope=logits_scope, reuse = True
            )

        softmax = tf.nn.softmax(logits, name="softmax")
        self.softmax = softmax
        val, idx = tf.nn.top_k(softmax, name="topk")

        #sentence = tf.Print(sentence, ["val", val[0], "idx", idx[0],sentence])

        sentence = tf.concat([sentence,idx[0]],0)
        self.output = sentence
        return [sentence, state]
    out = tf.while_loop(cond, body, [sentence, state],parallel_iterations=1,maximum_iterations=20,
                        name="loopx",shape_invariants=[tf.TensorShape([None]),tf.TensorShape([None,None])],
                            return_same_structure=True)

    return out
  def setup_inception_initializer(self):
    """Sets up the function to restore inception variables from checkpoint."""
    if self.mode != "inference":
      # Restore inception variables only.
      saver = tf.train.Saver(self.inception_variables)

      def restore_fn(sess):
        tf.logging.info("Restoring Inception variables from checkpoint file %s",
                        self.config.inception_checkpoint_file)
        saver.restore(sess, self.config.inception_checkpoint_file)

      self.init_fn = restore_fn

  def setup_global_step(self):
    """Sets up the global step Tensor."""
    global_step = tf.Variable(
        initial_value=0,
        name="global_step",
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    self.global_step = global_step

  def build(self):
    """Creates all ops for training and evaluation."""
    self.build_inputs()
    self.build_image_embeddings()
    self.build_seq_embeddings()
    self.out = self.build_model()
    self.setup_inception_initializer()
    self.setup_global_step()

import os.path


import tensorflow as tf

# pylint: disable=unused-argument





FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "./data/modelf.ckpt-2000000",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "./data/word_counts.txt", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "./toinfer/*",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = ShowAndTellModel(ModelConfig(),mode='inference')
    restore_fn = model.build_graph_from_config(FLAGS.checkpoint_path)

  # Create the vocabulary.
  vocab = Vocabulary(FLAGS.vocab_file)

  filenames = []
  for file_pattern in FLAGS.input_files.split(","):
    filenames.extend(tf.gfile.Glob(file_pattern))
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), FLAGS.input_files)

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    # generator = CaptionGenerator(model, vocab)

    for filename in filenames:
      with tf.gfile.GFile(filename, "rb") as f:
        image = f.read()

        sess.run(model.sentence.initializer)
        beamed = sess.run(fetches=["loopx/Exit_1:0"],feed_dict={"image_bytes:0": image})
        #print(beamed)
        sentence = [vocab.id_to_word(w) for w in beamed[0]]
        sentence = " ".join(sentence)
        print(sentence[3:-4])
        #continue
        simple_save.simple_save(sess, "./saved_model/0010", {"image": model.image_feed}, {"caption": model.output})
        break



if __name__ == "__main__":
  tf.app.run()
