import nnvm
import tvm
import numpy as np

# Tensorflow imports
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from google.protobuf import text_format
from tensorflow.python.framework import graph_util
from tensorflow.python.ops import array_ops
dir(tf.contrib)

#np.set_printoptions(threshold=np.nan)
#np.set_printoptions(suppress=True)

###############################################################################
# Reader
# ---------------------------------------------
# Read the input data to form vocabulary
import collections
import os
def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().encode("utf-8").decode("utf-8").replace("\n", "<eos>").split()

def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))
  #for python 3.x
  #id_to_word = dict((v, k) for k, v in word_to_id.items())
  id_to_word = dict((v, k) for k, v in word_to_id.iteritems())

  return word_to_id, id_to_word


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None, prefix="ptb"):
  """Load PTB raw data from data directory "data_path".
  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.
  The PTB dataset comes from Tomas Mikolov's webpage:
  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.
  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, prefix + ".train.txt")
  valid_path = os.path.join(data_path, prefix + ".valid.txt")
  test_path = os.path.join(data_path, prefix + ".test.txt")

  word_to_id, id_2_word = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary, word_to_id, id_2_word


###############################################################################
# PTB model
# ---------------------------------------------
# PTB model object

import tensorflow as tf
flags = tf.flags
FLAGS = flags.FLAGS


def data_type():
  #return tf.float16 if FLAGS.use_fp16 else tf.float32
  return tf.float32

class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    lstm_cell = tf.contrib.rnn.LSTMBlockCell(size, forget_bias=0.0)
    #lstm_cell = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
    if is_training and config.keep_prob < 1:
      lstm_cell = tf.contrib.rnn.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

    self._initial_state = cell.zero_state(batch_size, data_type())
    print("self._initial_state:", self._initial_state)

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, self._input_data)
      print("##################embedding#####################", embedding)
    print("########inputs#############", inputs)
    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # inputs = [tf.squeeze(input_step, [1])
    #           for input_step in tf.split(1, num_steps, inputs)]
    # outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)
    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, size])
    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b
    self.sample = tf.multinomial(logits, 1, seed=1)
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(self._targets, [-1])],
        [tf.ones([batch_size * num_steps], dtype=data_type())])
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state
    if not is_training:
      self._output_probs = tf.nn.softmax(logits)
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    if config.optimizer == 'RMSPropOptimizer':
      optimizer = tf.train.RMSPropOptimizer(self._lr)
    elif config.optimizer == 'AdamOptimizer':
      optimizer = tf.train.AdamOptimizer()
    elif config.optimizer == 'MomentumOptimizer':
      optimizer = tf.train.MomentumOptimizer(self._lr, momentum=0.8, use_nesterov=True)
    else:
      optimizer = tf.train.GradientDescentOptimizer(self._lr)
    #optimizer = tf.train.RMSPropOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)
    self._output_probs = tf.nn.softmax(logits)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets


  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def output_probs(self):
    return self._output_probs


###############################################################################
# Vocabulary
# ---------------------------------------------
# Create vocabulary from the input data provided
raw_data = ptb_raw_data('/home/albin/work_shared/workspace/models/tutorials/rnn/ptb/data/simple-examples/data')
train_data, valid_data, test_data, vocabulary, word_to_id, id_to_word = raw_data
vocab_size = len(word_to_id)

###############################################################################
# PTB test model
# ---------------------------------------------
# Ceate the PTB test model for sampling
class SmallConfig(object):
  """Small config."""
  is_char_model = False
  optimizer = 'AdamOptimizer'
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 4
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000

def get_config():
  return SmallConfig()

config = get_config()
config.vocab_size = config.vocab_size if config.vocab_size < vocab_size else vocab_size
eval_config = get_config()
eval_config.vocab_size = eval_config.vocab_size if eval_config.vocab_size < vocab_size else vocab_size
eval_config.batch_size = batch_size = 1
eval_config.num_steps = num_steps = 1

initializer = tf.random_uniform_initializer(config.init_scale,
                                            config.init_scale)
print("initializer:", initializer)
with tf.variable_scope("Model", reuse=None, initializer=initializer):
    #mtest = PTBModel(is_training=False, config=eval_config)
    mtest = None

###############################################################################
# Restore Model
# ---------------------------------------------
# Restore the PTB pre-trained model checkpoints
def _parse_input_graph_proto(input_graph, input_binary):
  """Parser input tensorflow graph into GraphDef proto."""
  if not tf.gfile.Exists(input_graph):
    print("Input graph file '" + input_graph + "' does not exist!")
    return -1
  input_graph_def = graph_pb2.GraphDef()
  mode = "rb" if input_binary else "r"
  with tf.gfile.FastGFile(input_graph, mode) as f:
    if input_binary:
      input_graph_def.ParseFromString(f.read())
    else:
      text_format.Merge(f.read(), input_graph_def)
  return input_graph_def

#create graph from saved meta files.
meta_path = '/home/albin/work_shared/workspace/models/rnn_text_writer_block_init_pro/final_check/model.ckpt.meta' # Your .meta file
input_binary = False

with tf.Session() as sess:
    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess,tf.train.latest_checkpoint('/home/albin/work_shared/workspace/models/rnn_text_writer_block_init_pro/final_check/'))

    graph_def = _parse_input_graph_proto('/home/albin/work_shared/workspace/models/rnn_text_writer_block_init_pro/final_check/run_graph.pbtxt', input_binary)

    #['Model/multinomial/Multinomial'],
    #        ['Model/RNN/strided_slice'],
    #        ['Model/RNN/strided_slice_1'],
    #[Model/Softmax]
    #Model/RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell
    #Model/embedding_lookup
    final_graph_def = graph_util.convert_variables_to_constants(
        sess,
        graph_def,
        ['Model/Softmax'],
        variable_names_whitelist=None,
        variable_names_blacklist=None)
sym, params = nnvm.frontend.from_tensorflow(final_graph_def)


######################################################################
# Compile the model on NNVM
# ---------------------------------------------
# We should be familiar with the process right now.
import nnvm.compiler
target = 'llvm'
batch_size = 1
num_steps = 1
num_hidden = 200
num_layers = 2
input_shape = (batch_size, num_steps)
output_shape = (1, 10000)
print ("---- CALL NNVM COMPILATION ---- BEGIN")
shape_dict = {'Model/Placeholder': input_shape, '_param_in_state':(num_layers, 2, batch_size, num_hidden)}
dtype_dict = {'Model/Placeholder': 'int32', '_param_in_state':'float32'}
graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, dtype=dtype_dict, params=params)
print ("---- CALL NNVM COMPILATION ---- END")

lib.export_library("imagenet_tensorflow.so")
with open("imagenet_tensorflow.json", "w") as fo:
    fo.write(graph.json())
with open("imagenet_tensorflow.params", "wb") as fo:
    fo.write(nnvm.compiler.save_param_dict(params))
print(graph.ir())

######################################################################
# Execute on TVM
# ---------------------------------------------
# The process is no different from other example
from tvm.contrib import graph_runtime
ctx = tvm.cpu(0)
out_dtype = 'float32'
m = graph_runtime.create(graph, lib, ctx)

######################################################################
# Predition
# ---------------------------------------------
# Create sample rediction results

def do_sample(model, model, data, in_states, num_samples):
  """Sampled from the model"""
  samples = []
  state = in_states
  fetches = [model.final_state, model.sample]
  sample = None
  print("data input:", data)
  print("Number of samples:", num_samples)
  for x in data:
    print("data input item:", x)
    word_id = np.full((batch_size, num_steps), x, dtype="int32")
    print("data input word_id:", word_id)
    m.set_input('Model/Placeholder', tvm.nd.array(word_id.astype("int32")))
    m.set_input('_param_in_state', tvm.nd.array(state.astype("float32")))
    m.set_input(**params)
    print("Sample first run")
    m.run()
    tvm_output = m.get_output(0, tvm.nd.empty(output_shape, out_dtype)).asnumpy()
    print("Output Softmax:", tvm_output)
    sample = pick_from_weight(tvm_output[0])
    print("Sample first:", sample)

  if sample is not None:
    samples.append(sample)
  else:
    samples.append(0)
  print("Samples temp: %s" % pretty_print(samples, False, id_to_word))
  print("samples numbers:", [x for x in samples])
  return samples

def pretty_print(items, is_char_model, id2word):
  if not is_char_model:
    return ' '.join([id2word[x] for x in items])
  else:
    return ''.join([id2word[x] for x in items]).replace('_', ' ')


# set inputs
word_id=np.full((batch_size, num_steps), 60, dtype="int32")
in_state = np.full((num_layers, 2, batch_size, num_hidden),0, dtype="float32")
m.set_input('Model/Placeholder', tvm.nd.array(word_id.astype("int32")))
m.set_input('_param_in_state', tvm.nd.array(in_state.astype("float32")))
m.set_input(**params)
# execute
#input_dict= {'Model/Placeholder':[[60]],}
#m.run(**input_dict)
m.run()

# get outputs
tvm_output = m.get_output(0, tvm.nd.empty(output_shape, out_dtype)).asnumpy()
print("Output strided slice:", tvm_output)


###############################################################################
# Input words
# ---------------------------------------------
# The input data provide the context to predict next word
in_state = np.full((num_layers, 2, batch_size, num_hidden), 0, dtype="float32")
while.True:
    inpt = raw_input("Enter your sample prefix: ")
    cnt = int(raw_input("Sample size: "))
    #python 3.x
    #inpt = input("Enter your sample prefix: ")
    #cnt = int(input("Sample size: "))
    print("Seed: %s" % pretty_print([word_to_id[x] for x in seed_for_sample], False, id_to_word))
    print("Sample: %s" % pretty_print(do_sample(m, mtest, [word_to_id[word] for word in seed_for_sample],
                                                  in_state, cnt), False, id_to_word))

###############################################################################
# Input words
# ---------------------------------------------
# The input data provide the context to predict next word
  '''while True:
  inpt = raw_input("Enter your sample prefix: ")
  cnt = int(raw_input("Sample size: "))
  #python 3.x
  #inpt = input("Enter your sample prefix: ")
  #cnt = int(input("Sample size: "))'''
  '''if config.is_char_model:
    seed_for_sample = [c for c in inpt.replace(' ', '_')]
  else:
    seed_for_sample = inpt.split()'''
  '''seed_for_sample = inpt.split()
  print("Seed: %s" % pretty_print([word_to_id[x] for x in seed_for_sample], False, id_to_word))
  print("Sample: %s" % pretty_print(do_sample(sess, mtest, [word_to_id[word] for word in seed_for_sample],
                                              cnt), False, id_to_word))'''
