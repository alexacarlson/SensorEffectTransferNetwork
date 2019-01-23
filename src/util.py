import tensorflow as tf
from numpy import *
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '{}/cityscapesScripts'.format(parentdir))
from cityscapesscripts.helpers.labels import labels as CITYSCAPES_LABELS


try:
    vgg_weights = load('/mnt/ngv/pretrained-networks/tensorflow-vgg/vgg16.npy', encoding='latin1').item()
except:
    raise FileNotFoundError('Download `vgg16.npy` from https://github.com/machrisaa/tensorflow-vgg first.')

num_classes = 0
cmap_trainId2color = zeros((256, 3), dtype=uint8)
cmap_trainId2id = zeros_like(cmap_trainId2color)
for l in CITYSCAPES_LABELS:
    if not l.ignoreInEval:
        num_classes += 1
        cmap_trainId2color[l.trainId, :] = l.color
        cmap_trainId2id[l.trainId, :] = l.id

cmap_trainId2color.reshape(cmap_trainId2color.size)
cmap_trainId2id.reshape(cmap_trainId2id.size)


def get_bilinear_kernel(ksize):
    f = ceil(ksize / 2)
    c = (2 * f - 1 - f % 2) / 2 / f
    bilinear = zeros([ksize, ksize])
    for x in range(ksize):
        for y in range(ksize):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    kernel = zeros([ksize, ksize, num_classes, num_classes])
    for i in range(num_classes):
        kernel[:, :, i, i] = bilinear
    return kernel


def max_pool(x, name):
    return tf.layers.max_pooling2d(x, 2, 2, padding='same', name=name)


def conv2d_relu_r(x, name):
    bias = vgg_weights[name][1]
    if 'conv' in name:
        kernel = vgg_weights[name][0]
    elif name == 'fc6':
        kernel = vgg_weights[name][0].reshape([7, 7, 512, 4096])
    elif name == 'fc7':
        kernel = vgg_weights[name][0].reshape([1, 1, 4096, 4096])

    with tf.variable_scope(name):
        x = tf.layers.conv2d(x, kernel.shape[-1], kernel.shape[:2],
                             padding='same', name='conv2d',
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.001, dtype=tf.float32),
                             bias_initializer=tf.constant_initializer(bias))
        return tf.nn.relu(x, name='relu')

def conv2d_relu2(x, name, training=True):
    bias = vgg_weights[name][1]
    if 'conv' in name:
        kernel = vgg_weights[name][0]
    elif name == 'fc6':
        kernel = vgg_weights[name][0].reshape([7, 7, 512, 4096])
    elif name == 'fc7':
        kernel = vgg_weights[name][0].reshape([1, 1, 4096, 4096])

    with tf.variable_scope(name):
        #tf.layers.conv2d(inputs, filters, kernel_size, strides=(1, 1), padding='valid', data_format='channels_last',)
        x = tf.layers.conv2d(x, kernel.shape[-1], kernel.shape[:2],
                             padding='same', name='conv2d',
                             kernel_initializer=tf.constant_initializer(kernel),
                             bias_initializer=tf.constant_initializer(bias),
                            trainable = training)
        return tf.nn.relu(x, name='relu')

def conv2d_relu(x, name):
    bias = vgg_weights[name][1]
    if 'conv' in name:
        kernel = vgg_weights[name][0]
    elif name == 'fc6':
        kernel = vgg_weights[name][0].reshape([7, 7, 512, 4096])
    elif name == 'fc7':
        kernel = vgg_weights[name][0].reshape([1, 1, 4096, 4096])

    with tf.variable_scope(name):
        #tf.layers.conv2d(inputs, filters, kernel_size, strides=(1, 1), padding='valid', data_format='channels_last',)
        x = tf.layers.conv2d(x, kernel.shape[-1], kernel.shape[:2],
                             padding='same', name='conv2d',
                             kernel_initializer=tf.constant_initializer(kernel),
                             bias_initializer=tf.constant_initializer(bias))
        return tf.nn.relu(x, name='relu')

def conv2d_relu_w(x, name, weights_dict):
    bias = weights_dict[name][1]
    if 'conv' in name:
        kernel = weights_dict[name][0]
    elif name == 'fc6':
        kernel = weights_dict[name][0].reshape([7, 7, 512, 4096])
    elif name == 'fc7':
        kernel = weights_dict[name][0].reshape([1, 1, 4096, 4096])

    with tf.variable_scope(name):
        x = tf.layers.conv2d(x, kernel.shape[-1], kernel.shape[:2],
                             padding='same', name='conv2d',
                             kernel_initializer=tf.constant_initializer(kernel),
                             bias_initializer=tf.constant_initializer(bias))
        return tf.nn.relu(x, name='relu')

def conv2d_w(x, name, weights_dict):
    bias = weights_dict[name][1]
    if 'conv' in name:
        kernel = weights_dict[name][0]
    elif name == 'fc6':
        kernel = weights_dict[name][0].reshape([7, 7, 512, 4096])
    elif name == 'fc7':
        kernel = weights_dict[name][0].reshape([1, 1, 4096, 4096])

    with tf.variable_scope(name):
        x = tf.layers.conv2d(x, kernel.shape[-1], kernel.shape[:2],
                             padding='same', name='conv2d',
                             kernel_initializer=tf.constant_initializer(kernel),
                             bias_initializer=tf.constant_initializer(bias))
        return x

def upscore(x, stride, name):
    kernel = get_bilinear_kernel(stride * 2)
    with tf.variable_scope(name):
        x = tf.layers.conv2d_transpose(x, num_classes, stride * 2, strides=stride,
                                       padding='same', name='conv2d_transpose', use_bias=False,
                                       kernel_initializer=tf.constant_initializer(kernel))
        return x

def upscore_w(x, stride, name, weights_dict):
    #kernel = get_bilinear_kernel(stride * 2)
    kernel = weights_dict[name][0]
    with tf.variable_scope(name):
        x = tf.layers.conv2d_transpose(x, num_classes, stride * 2, strides=stride,
                                       padding='same', name='conv2d_transpose', use_bias=False,
                                       kernel_initializer=tf.constant_initializer(kernel))
        return x


def cross_entropy(logits, labels, name='xent_loss'):
    with tf.variable_scope(name):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
