from keras.models import Model
from keras.layers import Input, Conv2D, add, BatchNormalization, Activation
from MosaicLayer import MosaicLayer

import numpy as np

class nn_model:

    def __init__(self):

        # Hyperpameters

        # Input / output expected image configuration
        self.image_shape = (16,16)
        self.image_channels = 3

        # Depth of the network
        self.num_conv_layers = 4

        # Number of convolutional filters learned per layer
        self.num_filters = 64

        # Shape of the convolutional kernels
        self.conv_kernel_size = (3,3)

    def setup_model(self):
        # Build the network graph.

        rgb_in = Input(shape=(self.image_shape[0], self.image_shape[1], self.image_channels))

        # Simulated image mosaic, with embedded color filter array data.
        mosaic_fcn = MosaicLayer(output_dim=(None, self.image_shape[0], self.image_shape[1], self.image_channels))
        mosaiced_img = mosaic_fcn(rgb_in)

        # Deep residual convolutional net of arbitrary depth.
        # Each layer consists of:
        #   - convolution;
        #   - batch normalization;
        #   - SELU activation;
        #   - Sum (residual learning -  https://arxiv.org/abs/1512.03385)

        conv_layers = []
        sum_layers = []

        # First convolutional layer does not have a skip connection to the input, because dimensions are different:

        first_conv = Conv2D(self.num_filters,
                            self.conv_kernel_size,
                            strides=(1, 1),
                            padding='same',
                            kernel_initializer='lecun_normal',
                            data_format="channels_last",
                            name=('Conv_00'))(mosaiced_img)

        last_input = first_conv
        for i in range(1,self.num_conv_layers):
            c = Conv2D(self.num_filters,
                       self.conv_kernel_size,
                       strides=(1,1),
                       padding='same',
                       kernel_initializer='lecun_normal',
                       data_format="channels_last",
                       name=('Conv_%02d' % i))(last_input)

            n = BatchNormalization(axis=3)(c)  # 3rd axis is image channels

            a = Activation('selu')(n)

            s = add([last_input, a])

            last_input = s

        # final 1x1 convolution to reduce to desired number of image channels
        rgb_out = Conv2D(filters=self.image_channels,
                         kernel_size=(1,1),
                         strides=(1,1))(last_input)

        model = Model(rgb_in, rgb_out)

        return model
