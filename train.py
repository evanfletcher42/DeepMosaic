import keras.callbacks
import numpy as np

import scipy.misc

import nn_model
from create_training_data import create_training_data

# ==================== Hyperparameters ====================

train_data_path = "data/Flickr500"
validation_data_path = "data/Validation"

# There are 490 images in the training set (the Flickr500 set, minus 10 images separated as the validation set)
# 35 is a reasonable small-ish divisor into 490
batch_size = 1

epochs = 5000

# validation_split = 0.005
validation_split = 0

np.random.seed(1337)

# =========== Initialize NN model object ==========
# This contains all the parameters about the network and a bunch of useful functions.
nn_m = nn_model.nn_model()

# ==================== Keras callbacks ====================
tb_cb = keras.callbacks.TensorBoard(log_dir='./logs',
                                    histogram_freq=5,
                                    batch_size=batch_size,
                                    write_graph=True,
                                    write_grads=False,
                                    write_images=True,
                                    embeddings_freq=0,
                                    embeddings_layer_names=None,
                                    embeddings_metadata=None,
                                    embeddings_data=None)

# Save the model every so often to prevent cardiac arrest.
ckpt_cb = keras.callbacks.ModelCheckpoint('checkpoints\\weights.{epoch:02d}.hdf5',
                                          # monitor='val_loss',
                                          verbose=0,
                                          save_best_only=False,
                                          save_weights_only=False,
                                          mode='auto',
                                          period=5)


# Periodically save out the learned color filter array as an image.
class DrawMosaicCallback(keras.callbacks.Callback):
    def __init__(self):
        super(DrawMosaicCallback, self).__init__()
        self.batches_seen = 0

    def on_batch_end(self, batch, logs=None):
        self.batches_seen += 1

        if self.batches_seen % 100 == 1:
            # Write out the weights of the mosaic layer as an RGB image.
            mosaic_weights = np.array(self.model.layers[1].get_weights())
            scipy.misc.toimage(mosaic_weights[0], cmin=0.0, cmax=1.0).save("mosaic/mosaic_%d.png" % self.batches_seen)

dm_cb = DrawMosaicCallback()

# ==================== Network definition ====================

model = nn_m.setup_model()

# ==================== Data preparation ====================

# Note: As currently programmed, the model is an autoencoder;
# it finds the best encoding (mosaic function) and decoding (demosaic conv net) to exactly reproduce the input.
# TODO: Break this assumption by introducing simulated image sensor noise to the dataset.

dataset = create_training_data(nn_m.image_shape, data_path=train_data_path)
val_set = create_training_data(nn_m.image_shape, data_path=validation_data_path)

print("Dataset shape: " + str(dataset.shape))

# ==================== Training ====================

model.compile(optimizer='adam',
              loss="mean_squared_error")

model.fit(dataset, dataset,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=[val_set, val_set],
          # callbacks=[tb_cb, ckpt_cb, dm_cb],  # enable this to periodically draw the learned CFA.
          callbacks=[tb_cb, ckpt_cb]
          )
