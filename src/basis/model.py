"""DeepLabMini model architecture.

Ported unchanged from ``codes_training/training/models/mapping/deeplab.py``
in the full CrevaU-Net repository. This is the exact architecture the
released weights were trained with (a DeepLabv3+-style decoder on top of a
ResNet-50 encoder, adapted from
https://medium.com/@itberrios6/deeplabv3-c0c8c93d25a4); do not change it
unless you are retraining the model with matching weights.
"""
import tensorflow as tf

from .layers import ConvBatchNormAct, ConvBatchNormAct_x2


def _dilated_spatial_pyramid_pooling(dspp_input, kernel_size, dropout):
    dims = dspp_input.shape
    x = tf.keras.layers.AveragePooling2D(pool_size=(dims[1], dims[2]))(dspp_input)
    x = ConvBatchNormAct(n_filters=128, kernel_size=kernel_size)(x)
    out_pool = tf.keras.layers.UpSampling2D(
        size=(dims[1] // x.shape[1], dims[2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = ConvBatchNormAct(n_filters=128, kernel_size=kernel_size, dilation_rate=1)(dspp_input)
    out_1_2 = ConvBatchNormAct(n_filters=128, kernel_size=3, dilation_rate=1)(dspp_input)
    out_6 = ConvBatchNormAct(n_filters=128, kernel_size=3, dilation_rate=6)(dspp_input)
    out_12 = ConvBatchNormAct(n_filters=128, kernel_size=3, dilation_rate=12)(dspp_input)
    out_18 = ConvBatchNormAct(n_filters=128, kernel_size=3, dilation_rate=18)(dspp_input)
    out_24 = ConvBatchNormAct(n_filters=128, kernel_size=3, dilation_rate=24)(dspp_input)

    x = tf.keras.layers.Concatenate(axis=-1)([out_pool, out_1, out_1_2, out_6, out_12, out_18, out_24])
    return ConvBatchNormAct(n_filters=128, kernel_size=kernel_size, spatial_dropout=dropout)(x)


def build_deeplab_mini(
    input_shape, n_classes, kernel_size=1, dropout=0, last_activation="softmax", name="DeepLabMini",
):
    """Build the DeepLabMini segmentation model.

    Parameters
    ----------
    input_shape : tuple
        ``(patch_size, patch_size, n_features)``.
    n_classes : int
        Number of output classes (background + fracture class(es)).
    kernel_size, dropout : int, float
        Must match the values used to train the weights you intend to load
        (they change the shape/presence of some layers). ``dropout`` has no
        effect on inference results (it is inactive outside of training)
        but changing ``kernel_size`` changes convolution kernel shapes.
    """
    inputs = tf.keras.layers.Input(input_shape, name="features")
    dims = inputs.shape

    resnet50 = tf.keras.applications.ResNet50(weights=None, include_top=False, input_tensor=inputs)
    x = resnet50.get_layer("conv3_block4_2_relu").output
    x = _dilated_spatial_pyramid_pooling(x, kernel_size, dropout)

    input_a = tf.keras.layers.UpSampling2D(
        size=(dims[1] // 4 // x.shape[1], dims[2] // 4 // x.shape[2]), interpolation="bilinear",
    )(x)

    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = ConvBatchNormAct(n_filters=48, kernel_size=kernel_size, spatial_dropout=dropout)(input_b)

    x = tf.keras.layers.Concatenate(axis=-1)([input_a, input_b])
    x = ConvBatchNormAct_x2(n_filters=64, spatial_dropout=dropout)(x)
    x = tf.keras.layers.UpSampling2D(
        size=(dims[1] // x.shape[1], dims[2] // x.shape[2]), interpolation="bilinear",
    )(x)

    outputs = tf.keras.layers.Conv2D(n_classes, 1, activation=last_activation)(x)
    return tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name)
