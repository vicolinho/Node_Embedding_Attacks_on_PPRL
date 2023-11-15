import numpy as np
import pandas as pd
import tensorflow as tf
from stellargraph.data import UnsupervisedSampler
from stellargraph.layer import GraphSAGE, link_classification, GCN, DeepGraphInfomax
from stellargraph.mapper import GraphSAGELinkGenerator, GraphSAGENodeGenerator, GraphWaveGenerator, \
    FullBatchNodeGenerator, CorruptedGenerator
from stellargraph.utils import plot_history
from tensorflow import keras
from tensorflow.python.keras.callbacks import EarlyStopping

from attack.graphwave_ext import graphwave_ext
from classes.embedding_results import Embedding_results


def generate_node_embeddings_graphsage(G, graphsage_settings, scaler):
    # https://stellargraph.readthedocs.io/en/stable/demos/embeddings/graphsage-unsupervised-sampler-embeddings.html
    nodes = list(G.nodes())
    number_of_walks = graphsage_settings.number_of_walks
    length = graphsage_settings.length
    unsupervised_samples = UnsupervisedSampler(
        G, nodes=nodes, length=length, number_of_walks=number_of_walks
    )
    batch_size = graphsage_settings.batch_size
    epochs = graphsage_settings.epochs
    num_samples = graphsage_settings.num_samples
    generator = GraphSAGELinkGenerator(G, batch_size, num_samples)
    train_gen = generator.flow(unsupervised_samples)
    layer_sizes = graphsage_settings.layers
    graphsage = GraphSAGE(
        layer_sizes=layer_sizes, generator=generator, bias=True, dropout=0.0, normalize="l2"
    )
    # Build the model and expose input and output sockets of graphsage, for node pair inputs:
    x_inp, x_out = graphsage.in_out_tensors()
    prediction = link_classification(
        output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
    )(x_out)
    model = keras.Model(inputs=x_inp, outputs=prediction)

    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy],
    )
    history = model.fit(
        train_gen,
        epochs=epochs,
        verbose=1,
        use_multiprocessing=False,
        workers=4,
        shuffle=True,
    )
    x_inp_src = x_inp[0::2]
    x_out_src = x_out[0]
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
    node_gen = GraphSAGENodeGenerator(G, batch_size, num_samples).flow(nodes)
    node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)
    embeddings_transformed = normalize_embeddings(node_embeddings, scaler)
    return Embedding_results(embeddings_transformed, nodes, str(graphsage_settings), graphsage_settings)


def generate_node_embeddings_graphwave(G, graphwave_settings, scaler, external_lib):
    """
    calculates node embeddings either with a custom library to be imported or if not found with StellarGraph
    StellarGraph one doesn't support edge weights
    :param G:
    :param graphwave_settings:
    :param scaler:
    :return:
    """
    if external_lib:
        return generate_node_embeddings_graphwave_lib(G, graphwave_settings, scaler)
    else:
        graphwave_settings.set_label_for_stellargraph_func()
        return generate_node_embeddings_graphwave_sg(G, graphwave_settings, scaler)

def generate_node_embeddings_graphwave_sg(G, graphwave_settings, scaler):
    sample_points = np.linspace(
        0, graphwave_settings.sample_p_max_val, graphwave_settings.no_samples
    ).astype(np.float32)
    degree = graphwave_settings.order_approx
    scales = graphwave_settings.scales

    generator = GraphWaveGenerator(G, scales=scales, degree=degree)
    node_ids = G.nodes()
    embeddings_dataset = generator.flow(
        node_ids=node_ids, sample_points=sample_points, batch_size=1, repeat=False
    )

    embeddings = [x.numpy() for x in embeddings_dataset]
    embeddings = np.reshape(embeddings, (len(embeddings), len(embeddings[0][0])))
    embeddings_transformed = normalize_embeddings(embeddings, scaler)
    return Embedding_results(embeddings_transformed, node_ids, str(graphwave_settings), graphwave_settings)


def generate_node_embeddings_graphwave_lib(G, graphwave_settings, scaler):
    sample_points = np.linspace(
        0, graphwave_settings.sample_p_max_val, graphwave_settings.no_samples
    )
    if graphwave_settings.scales == ['auto']:
        chi, heat_print, taus = graphwave_ext.graphwave_alg(G, sample_points, order=graphwave_settings.order_approx)
        graphwave_settings.scales = taus
    else:
        chi, heat_print, taus = graphwave_ext.graphwave_alg(G, sample_points, taus=graphwave_settings.scales, order=graphwave_settings.order_approx)

    node_ids = list(G)
    embeddings_transformed = normalize_embeddings(chi, scaler)
    return Embedding_results(embeddings_transformed, node_ids, str(graphwave_settings), graphwave_settings)


def normalize_embeddings(embeddings, scaler):
    scaler.fit(embeddings)
    embeddings_transformed = scaler.transform(embeddings)
    return embeddings_transformed

def generate_node_embeddings_deepgraphinfomax(G, deepgraphinfomax_settings, scaler):
    fullbatch_generator = FullBatchNodeGenerator(G, sparse=False)
    gcn_model = GCN(layer_sizes=deepgraphinfomax_settings.layers, activations=deepgraphinfomax_settings.activations, generator=fullbatch_generator)
    corrupted_generator = CorruptedGenerator(fullbatch_generator)
    gen = corrupted_generator.flow(G.nodes())
    infomax = DeepGraphInfomax(gcn_model, corrupted_generator)
    x_in, x_out = infomax.in_out_tensors()

    model = keras.Model(inputs=x_in, outputs=x_out)
    model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer=keras.optimizers.Adam(lr=1e-3))
    epochs = deepgraphinfomax_settings.epochs
    es = EarlyStopping(monitor="loss", min_delta=0, patience=20)
    #history = model.fit(gen, epochs=epochs, verbose=1, callbacks=[es])
    history = model.fit(gen, epochs=epochs, verbose=1, callbacks=[es], batch_size=32)
    plot_history(history)
    x_emb_in, x_emb_out = gcn_model.in_out_tensors()

    # for full batch models, squeeze out the batch dim (which is 1)
    x_out = tf.squeeze(x_emb_out, axis=0)
    emb_model = keras.Model(inputs=x_emb_in, outputs=x_out)

    node_subjects = G.nodes()


    node_gen = fullbatch_generator.flow(node_subjects)
    embeddings = emb_model.predict(node_gen)
    embeddings = normalize_embeddings(embeddings, scaler)
    return Embedding_results(embeddings, node_subjects, str(deepgraphinfomax_settings), deepgraphinfomax_settings)


def just_features_embeddings(G, settings):
    embeddings = G.node_features()
    df = pd.DataFrame(data=embeddings, index=G.nodes())
    embeddings = df.to_numpy()
    embeddings = normalize_embeddings(embeddings, settings.scaler)
    return Embedding_results(embeddings, df.index, "features")

def multiple_embeddings_to_one(embeddings_):
    """helper function: when analysing combination of embeddings you have to distinguish between both kinds
    to calculate weighted distances BUT for LSH you need concatenated embeddings"""

    if len(embeddings_) != 1:
        embeddings = np.concatenate((embeddings_[0], embeddings_[1]), axis=1)
        return embeddings
    else:
        return embeddings_[0]




