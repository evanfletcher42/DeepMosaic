from keras import backend as K
from keras.engine.topology import Layer
from keras.constraints import non_neg, max_norm
from keras.initializers import truncated_normal, constant
import numpy as np

class MosaicLayer(Layer):
    """Layer which simulates application of a trainable color filter array (CFA) to an input image.
    
    The color filter array is simulated as a per-pixel dot product over all channels, between an input pixel's channels 
    and a per-pixel weight vector directly representing a color in the CFA.  This is similar in principle to a 
    LocallyConnected2D layer over a 1x1 window producing a 1-channel output image.
    
    This layer then per-pixel multiplies the 1-channel dot-product result with the layer weights, resulting in an output 
    image with an equal number of channels as the input image.  This effectively embeds information about each pixel's
    CFA in the activations of this layer, but still loses information compared to the RGB input due to the dot product -
    i.e. for an input RGB pixel with values (0.500, 0.100, 0.200) and CFA value (0.800, 0.200, 0.100), the output is 
    (0.352, 0.088, 0.044), a vector which does not preserve the relative brightness of the input blue channel.  
    
    This embedding is done to communicate the nature of the CFA to later layers in the network, and is the key 
    innovation of this approach.  In previous work which attempts to learn the CFA (Syu et al, 2018), a separation of 
    each color plane - one per CFA color - is enforced in the network structure; the network learns to combine color 
    samples based on the location of data in N channels for N different color filters.  Such an approach is only 
    practical for small CFAs, which are repeated many times over the image in a fixed manual pattern (Syu et al, 2018).  
    
    This approach's per-pixel embedding of the CFA seeks to eliminate these manual design aspects of previous work.
    Instead of the number of possible filters and their repetitions being fixed and manually-selected prior to training,
    this approach can learn a unique color filter per-pixel, without any constraints on arrangement or uniqueness.
    
    Input: 4D tensor with shape:
    `(samples, rows, cols, channels)`
    
    Output: 4D tensor with shape:
    `(samples, rows, cols, channels)`
    
    """
    def __init__(self, output_dim, **kwargs):

        self.output_dim = output_dim
        super(MosaicLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create trainable weights for this layer.
        # This layer simulates a mosaic function by applying a per-pixel dot product on input channels.
        # The total number of weights is equal to the input image shape (img_width * img_height * img_channels)

        print("input_shape: " + str(input_shape))

        self.cfa = self.add_weight(name='colorFilter',
                                   shape=(input_shape[1], input_shape[2], input_shape[3]),
                                   initializer=constant(value=0.5),
                                   trainable=True,
                                   constraint=non_neg())

        super(MosaicLayer, self).build(input_shape)

    def call(self, x):
        dot_product = K.sum(x * self.cfa, axis=3, keepdims=True)  # shape: (batch, sample, rows, cols, 1)
        cfa_contrib = dot_product * self.cfa

        return cfa_contrib

    def compute_output_shape(self, input_shape):
        return input_shape
