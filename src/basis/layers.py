"""Custom Keras layers used by the DeepLabMini architecture.

Ported from the full training codebase
(``codes_training/training/layers/general/layers.py``), keeping only the
layers that DeepLabMini actually uses.
"""
import tensorflow as tf


class ConvBatchNormAct(tf.keras.layers.Layer):
    def __init__(
        self, n_filters, kernel_size=3, dilation_rate=1, use_bias=True, padding="same",
        activation=tf.keras.layers.LeakyReLU, spatial_dropout=0, **kwargs
    ):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(
            n_filters, kernel_size, dilation_rate=dilation_rate, use_bias=use_bias, padding=padding
        )
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = activation()
        self.spatial_dropout = spatial_dropout
        if spatial_dropout:
            self.dropout = tf.keras.layers.SpatialDropout2D(spatial_dropout)

    def call(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.act(y)
        if self.spatial_dropout:
            y = self.dropout(y)
        return y


class ConvBatchNormAct_x2(tf.keras.layers.Layer):
    def __init__(
        self, n_filters, kernel_size=3, dilation_rate=1, use_bias=True, padding="same",
        activation=tf.keras.layers.LeakyReLU, spatial_dropout=0, **kwargs
    ):
        super().__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            n_filters, kernel_size, dilation_rate=dilation_rate, use_bias=use_bias, padding=padding
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = activation()
        self.conv2 = tf.keras.layers.Conv2D(
            n_filters, kernel_size, dilation_rate=dilation_rate, use_bias=use_bias, padding=padding
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.act2 = activation()
        self.spatial_dropout = spatial_dropout
        if spatial_dropout:
            self.dropout = tf.keras.layers.SpatialDropout2D(spatial_dropout)

    def call(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act2(y)
        if self.spatial_dropout:
            y = self.dropout(y)
        return y
