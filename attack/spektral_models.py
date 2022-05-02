# https://github.com/danielegrattarola/spektral/blob/master/examples/other/node_clustering_mincut.py

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics.cluster import (
    completeness_score,
    homogeneity_score,
    v_measure_score,
)
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tqdm import tqdm

from spektral.datasets.citation import Cora
from spektral.layers.convolutional import GCSConv
from spektral.layers.pooling import MinCutPool
from spektral.utils.convolution import normalized_adjacency
from spektral.utils.sparse import sp_matrix_to_sp_tensor


@tf.function
def train_step(inputs):
    with tf.GradientTape() as tape:
        _, S_pool = model(inputs, training=True)
        loss = sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return model.losses[0], model.losses[1], S_pool


np.random.seed(1)
epochs = 5000  # Training iterations
lr = 5e-4  # Learning rate

################################################################################
# LOAD DATASET
################################################################################
dataset = Cora()
adj, x, _ = dataset[0].a, dataset[0].x, dataset[0].y
a_norm = normalized_adjacency(adj)
a_norm = sp_matrix_to_sp_tensor(a_norm)
F = dataset.n_node_features
n_clusters = 200

################################################################################
# MODEL
################################################################################
x_in = Input(shape=(F,), name="X_in")
a_in = Input(shape=(None,), name="A_in", sparse=True)

x_1 = GCSConv(16, activation="elu")([x_in, a_in])
x_1, a_1, s_1 = MinCutPool(n_clusters, return_selection=True)([x_1, a_in])

model = Model([x_in, a_in], [x_1, s_1])

######################################
encoded = layers.Dense(128, activation='relu')([x_in, a_in])
encoded = layers.Dense(64, activation='relu')(encoded)
encoded = layers.Dense(32, activation='relu')(encoded)

decoded = layers.Dense(64, activation='relu')(encoded)
decoded = layers.Dense(128, activation='relu')(decoded)
decoded = layers.Dense(784, activation='sigmoid')(decoded)

################################################################################
# TRAINING
################################################################################
# Setup
inputs = [x, a_norm]
opt = tf.keras.optimizers.Adam(learning_rate=lr)

# Fit model
loss_history = []
for _ in tqdm(range(epochs)):
    outs = train_step(inputs)
    outs = [o.numpy() for o in outs]
    s_out = np.argmax(outs[2], axis=-1)
_, s_out = model(inputs, training=False)
pass

