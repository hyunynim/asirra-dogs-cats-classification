from abc import abstractmethod
import tensorflow as tf
import numpy as np
from models.layers import conv_layer, max_pool, relu, fc_layer


class ConvNet(object):
    """
    Base class for Convolutional Neural Networks.
    """

    def __init__(self, image_shape, num_classes, **kwargs):
        """
        Model initializer.
        :param image_shape: Tuple, shape of input images (H, W, C).
        :param num_classes: Integer, number of classes.
        """
        self.X = tf.placeholder(tf.float32, [None])    # range: [0.0, 1.0]
        self.y = tf.placeholder(tf.float32, [None] + [num_classes])

        self.is_train = tf.placeholder(tf.bool)
        self.d = self._build_model(**kwargs)
        self.logits = self.d['logits']
        self.predict = self.d['pred']

    @abstractmethod
    def _build_model(self, **kwargs):
        """
        Model builder.
        This should be implemented.
        """
        pass

    @abstractmethod
    def _build_loss(self, **kwargs):
        """
        Build loss function for the model training.
        This should be implemented.
        """
        pass


class AlexNet(ConvNet):
    """
    AlexNet class.
    """

    def _build_model(self, **kwargs):
        """Model builder."""
        d = dict()    # Dictionary to save intermediate values returned from each layer.
        X_mean = kwargs.pop('image_mean')
        num_classes = int(self.y.get_shape()[-1])

        # input
        X_input = self.X - X_mean    # perform mean subtraction

        # conv1 - relu1 - pool1
        with tf.variable_scope('conv1'):
            d['conv1'] = conv_layer(X_input, 11, 4, 96, padding='VALID',
                                    weights_stddev=0.01, bias_value=0.0)
            print('conv1.shape', d['conv1'].get_shape().as_list())
        d['relu1'] = relu(d['conv1'])
        # (227, 227, 3) --> (55, 55, 96)
        d['pool1'] = max_pool(d['relu1'], 3, 2, padding='VALID')
        # (55, 55, 96) --> (27, 27, 96)
        print('pool1.shape', d['pool1'].get_shape().as_list())

        # conv2 - relu2 - pool2
        with tf.variable_scope('conv2'):
            d['conv2'] = conv_layer(d['pool1'], 5, 1, 256, padding='SAME',
                                    weights_stddev=0.01, bias_value=0.1)
            print('conv2.shape', d['conv2'].get_shape().as_list())
        d['relu2'] = relu(d['conv2'])
        # (27, 27, 96) --> (27, 27, 256)
        d['pool2'] = max_pool(d['relu2'], 3, 2, padding='VALID')
        # (27, 27, 256) --> (13, 13, 256)
        print('pool2.shape', d['pool2'].get_shape().as_list())

        # conv3 - relu3
        with tf.variable_scope('conv3'):
            d['conv3'] = conv_layer(d['pool2'], 3, 1, 384, padding='SAME',
                                    weights_stddev=0.01, bias_value=0.0)
            print('conv3.shape', d['conv3'].get_shape().as_list())
        d['relu3'] = relu(d['conv3'])
        # (13, 13, 256) --> (13, 13, 384)

        # conv4 - relu4
        with tf.variable_scope('conv4'):
            d['conv4'] = conv_layer(d['relu3'], 3, 1, 384, padding='SAME',
                                    weights_stddev=0.01, bias_value=0.1)
            print('conv4.shape', d['conv4'].get_shape().as_list())
        d['relu4'] = relu(d['conv4'])
        # (13, 13, 384) --> (13, 13, 384)

        # conv5 - relu5 - pool5
        with tf.variable_scope('conv5'):
            d['conv5'] = conv_layer(d['relu4'], 3, 1, 256, padding='SAME',
                                    weights_stddev=0.01, bias_value=0.1)
            print('conv5.shape', d['conv5'].get_shape().as_list())
        d['relu5'] = relu(d['conv5'])
        # (13, 13, 384) --> (13, 13, 256)
        d['pool5'] = max_pool(d['relu5'], 3, 2, padding='VALID')
        # (13, 13, 256) --> (6, 6, 256)
        print('pool5.shape', d['pool5'].get_shape().as_list())

        # Flatten feature maps
        f_dim = int(np.prod(d['pool5'].get_shape()[1:]))
        f_emb = tf.reshape(d['pool5'], [-1, f_dim])
        # (6, 6, 256) --> (9216)

        # fc6
        with tf.variable_scope('fc6'):
            d['fc6'] = fc_layer(f_emb, 4096,
                                weights_stddev=0.005, bias_value=0.1)
        d['relu6'] = relu(d['fc6'])
        # (9216) --> (4096)
        print('relu6.shape', d['relu6'].get_shape().as_list())
        # TODO: Add dropout layer

        # fc7
        with tf.variable_scope('fc7'):
            d['fc7'] = fc_layer(d['relu6'], 4096,
                                weights_stddev=0.005, bias_value=0.1)
        d['relu7'] = relu(d['fc7'])
        # (4096) --> (4096)
        print('relu7.shape', d['relu7'].get_shape().as_list())
        # TODO: Add dropout layer

        # fc8
        with tf.variable_scope('fc8'):
            d['logits'] = fc_layer(d['relu7'], num_classes,
                                weights_stddev=0.01, bias_value=0.0)
        # (4096) --> (num_classes)

        d['pred'] = tf.nn.softmax(d['logits'])

        return d

    def _build_loss(self, **kwargs):
        """Evaluate loss for the model."""
        weight_decay = kwargs.pop('weight_decay', 0.0005)
        variables = tf.trainable_variables()
        l2_reg_loss = tf.add_n([tf.nn.l2_loss(var) for var in variables])

        # Softmax cross-entropy loss function
        softmax_losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
        softmax_loss = tf.reduce_sum(softmax_losses)

        return softmax_loss + weight_decay*l2_reg_loss

