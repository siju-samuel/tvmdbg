"""
Compile Tensorflow Models
=====================
This article is a test script to test tensorflow models with NNVM.
All the required models and libraries will be downloaded from the internet
by the script.
"""

import tvm
import nnvm
import numpy as np
import tensorflow as tf


def test_forward(layer, ip_shape, ip_tensor, ip_name, out_shape, dtype):
    '''Test network with given input image on both tensorflow and tvm'''

    def load_image(ip_shape):
        shape_size = 1
        for posi in range (len(ip_shape)):
            shape_size = shape_size * ip_shape[posi]
        img = np.arange(shape_size, dtype=dtype).reshape(ip_shape)
        return img

    def get_tensorflow_output(layer, ip_tensor, img):
        with tf.Session() as sess:
            feed_dict={}
            for i in range (len(ip_tensor)):
                feed_dict.setdefault(ip_tensor[i],img)
            tf_out = sess.run(layer, feed_dict)
            #tf.train.write_graph(sess.graph.as_graph_def(add_shapes=True), ".", "tf_reshape_test.pbtxt")
            graph_def = sess.graph.as_graph_def(add_shapes=True)
            return graph_def, tf_out

    def get_tvm_output(graph_def, ip_name, img, out_shape):
        '''Compute TVM output'''

        sym, params = nnvm.frontend.from_tensorflow(graph_def)

        target = 'llvm'
        shape_dict = {}
        for i in range (len(ip_name)):
            shape_dict.setdefault(ip_name[i],img.shape)

        #with nnvm.compiler.build_config(opt_level=2):
        graph, library, params = nnvm.compiler.build(sym, target, shape_dict, dtype, params=params)

        from tvm.contrib import graph_runtime
        ctx = tvm.cpu(0)
        m = graph_runtime.create(graph, library, ctx)

        # set inputs
        for i in range (len(ip_name)):
            m.set_input(ip_name[i], tvm.nd.array(img.astype(dtype)))

        m.set_input(**params)
        m.run()
        # get outputs
        tvm_out = m.get_output(0, tvm.nd.empty(out_shape, dtype)).asnumpy()
        return tvm_out


    img = load_image(ip_shape)
    graph_def, tf_output = get_tensorflow_output(layer, ip_tensor, img)
    print("tf_output:", tf_output.shape, "\n", tf_output)
    tvm_out = get_tvm_output(graph_def, ip_name, img, out_shape)
    print("tvm_out:", tvm_out.shape, "\n", tvm_out)
    np.testing.assert_allclose(tf_output, tvm_out, rtol=1e-3, atol=1e-3)


def test_forward_rnn(fetch, ip_shape, ip_tensor, ip_name, out_shape, dtype):
    '''RNN test network with given input image on both tensorflow and tvm'''

    def load_image(ip_shape):
        shape_size = 1
        for posi in range (len(ip_shape)):
            shape_size = shape_size * ip_shape[posi]
        img = np.arange(shape_size, dtype=dtype).reshape(ip_shape)
        return img

    def get_tensorflow_output(fetch, ip_tensor, img):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict={}
            feed_dict.setdefault(ip_tensor,img)
            tf_out = sess.run(fetch, feed_dict)
            tf.train.write_graph(sess.graph.as_graph_def(add_shapes=True), ".", "tf_lstm_test.pbtxt")
            graph_def = sess.graph.as_graph_def(add_shapes=True)
            return graph_def, tf_out

    def get_tvm_output(graph_def, ip_name, imgs, out_shape):
        '''Compute TVM output'''

        sym, params = nnvm.frontend.from_tensorflow(graph_def)

        target = 'llvm'
        shape_dict = {}
        for i in range (len(ip_name)):
            shape_dict.setdefault(ip_name[i],img[i].shape)

        #with nnvm.compiler.build_config(opt_level=2):
        graph, library, params = nnvm.compiler.build(sym, target, shape_dict, dtype, params=params)

        from tvm.contrib import graph_runtime
        ctx = tvm.cpu(0)
        m = graph_runtime.create(graph, library, ctx)

        # set inputs
        for i in range (len(ip_name)):
            m.set_input(ip_name[i], tvm.nd.array(img[i].astype(dtype)))

        m.set_input(**params)
        m.run()
        # get outputs
        tvm_out = m.get_output(0, tvm.nd.empty(out_shape, dtype)).asnumpy()
        return tvm_out


    print("ip_shape:", ip_shape)
    img = load_image(ip_shape)
    graph_def, tf_output = get_tensorflow_output(fetch, ip_tensor, img)
    #print("tf_output:", tf_output.shape, "\n", tf_output)
    print("tf_output:", tf_output)
    tvm_out = get_tvm_output(graph_def, ip_name, img, out_shape)
    #print("tvm_out:", tvm_out.shape, "\n", tvm_out)
    print("tvm_out:", tvm_out)
    np.testing.assert_allclose(tf_output, tvm_out, rtol=1e-3, atol=1e-3)


def test_fill(ip_shape, fill_value, dtype):
    shape_length = 1
    for posi in range (len(ip_shape)):
        shape_length = shape_length * ip_shape[posi]
    tf.reset_default_graph()
    x = tf.placeholder(dtype, shape=ip_shape, name='x')
    y = tf.fill(ip_shape, fill_value)
    test_forward(y, ip_shape, [x], [], ip_shape, dtype)

def test_stack(ip_shape, axis, op_shape, num_ip, dtype):
    tf.reset_default_graph()
    input_names = [("input{:d}".format(i+1)) for i in range(num_ip)]
    inputs = [tf.placeholder(dtype, ip_shape, name=name) for name in input_names]
    output= tf.stack(inputs, axis=axis)
    test_forward(output, ip_shape, inputs, input_names, op_shape, dtype)

def test_reshape(ip_shape, tf_op_shape, op_shape, dtype):
    tf.reset_default_graph()
    x = tf.placeholder(dtype, ip_shape, name='x')
    y = tf.reshape(x, tf_op_shape, name="test_reshape")
    test_forward(y, ip_shape, [x], ['x'], op_shape, dtype)

def test_lstm_cell(batch_size, input_size, num_hidden,
            num_layers, num_times, forget_bias, dtype):
    tf.reset_default_graph()
    hidden_layers = 4 * num_hidden
    dtype='float32'
    vocab_size = 10000

    def _prepare_input():
        ip_shape = (batch_size, input_size)
        op_shape = (1, vocab_size)
        Xi2h = tf.placeholder(dtype, ip_shape, name="Xi2h")
        return Xi2h, ip_shape, op_shape

    Xi2h, ip_shape, op_shape = _prepare_input()
    lstm_cell = tf.contrib.rnn.LSTMBlockCell(num_hidden, forget_bias=forget_bias)
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)
    _initial_state = cell.zero_state(batch_size, dtype)
    state = _initial_state
    (cell_output, state) = cell(Xi2h, state)
    output = tf.reshape(tf.concat(axis=1, values=cell_output), [-1, num_hidden])
    softmax_w = tf.get_variable(
        "softmax_w", [num_hidden, vocab_size], dtype=dtype)
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=dtype)
    logits = tf.matmul(output, softmax_w) + softmax_b
    _output_probs = tf.nn.softmax(logits)
    fetch = [_output_probs]
    test_forward_rnn(fetch, ip_shape, Xi2h,
                     ['xi1'], op_shape, dtype)

def test_forward_fill():
    '''test fill layer'''
    test_fill((10,), 7, 'int32')
    test_fill((3,4,5), 3.0, 'float32')
    test_fill((4,), 3.14, 'float32')
    test_fill((1,5,9), 3, 'int32')

def test_forward_reshape():
    '''test reshape layer'''
    test_reshape((4,), [2, -1], [2,2], 'float32')
    test_reshape([1, 4], [2, -1], [2,2], 'int32')
    test_reshape([1, 2, 3, 3], [2, -1], [2,9], 'float32')
    test_reshape([1, 3, 2, 2], [-1, 3], [4,3], 'int32')

def test_forward_stack():
    '''test stack layer'''
    test_stack((1,), 0, (1,1), 1, 'float32')
    test_stack((1,), 0, (2,1), 2, 'int32')
    test_stack((1,3), 0, (2,1,3), 2, 'float32')
    test_stack((2,3), 1, (2,3,3), 3, 'int32')
    test_stack((1,2,3), 0, (3,1,2,3), 3, 'float32')
    test_stack((1,2,3), 3, (1,2,3,6), 6, 'int32')

def test_forward_lstm():
    '''test LSTM block cell'''
    test_lstm_cell(1, 2, 2, 2, 2, 1.0, 'float32')

if __name__ == '__main__':
    #test_forward_reshape()
    #test_forward_stack()
    #test_forward_fill()
    test_forward_lstm()
