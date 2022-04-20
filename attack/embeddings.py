import networkx as nx
import numpy as np
import stellargraph.core.convert
import tensorflow as tf
from gensim.models import Word2Vec
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk, UnsupervisedSampler
from stellargraph.layer import GraphSAGE, link_classification, GCN, DeepGraphInfomax
from stellargraph.mapper import GraphSAGELinkGenerator, GraphSAGENodeGenerator, GraphWaveGenerator, \
    FullBatchNodeGenerator, CorruptedGenerator
from stellargraph.utils import plot_history
from tensorflow import keras
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizer_v2.adam import Adam


def generate_node_embeddings_node2vec(graph): # not useful only includes node ids not structures or similarities
    # https://stellargraph.readthedocs.io/en/stable/demos/embeddings/node2vec-embeddings.html
    rw = BiasedRandomWalk(graph)
    node_list = list(graph.nodes())
    walks = rw.run(
        nodes=node_list,  # root nodes
        length=10,  # maximum length of a random walk 100
        n=10,  # number of random walks per root node 10
        p=1.0,  # Defines (unormalised) probability, 1/p, of returning to source node 0.5
        q=2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node 2.0
    )
    print("Number of random walks: {}".format(len(walks)))

    str_walks = [[str(n) for n in walk] for walk in walks]
    model = Word2Vec(str_walks, vector_size=128, window=5, min_count=0, sg=1, workers=2, epochs=1)
    node_ids = model.wv.index_to_key  # list of node IDs
    node_embeddings = (
        model.wv.vectors
    )  # numpy.ndarray of size number of nodes times embeddings dimensionality

    # TESTING PURPOSES:
    return node_embeddings, node_ids


def add_node_features_to_embeddings(node_embeddings, node_ids, node_weights, count):
    new_size = (np.shape(node_embeddings)[0], np.shape(node_embeddings)[1] + count)
    node_embeddings_new = np.zeros(new_size)
    for i in range(0, len(node_embeddings)):
        weight = node_weights.loc[str(node_ids[0])][0]
        node_embeddings_new[i] = np.append(node_embeddings[i], (count * [weight]))
    return node_embeddings_new


def generate_node_embeddings_graphsage(G, learning_G = None):
    # https://stellargraph.readthedocs.io/en/stable/demos/embeddings/graphsage-unsupervised-sampler-embeddings.html
    if learning_G == None:
        learning_G = G
    learning_nodes = list(learning_G.nodes())
    number_of_walks = 1
    length = 5
    unsupervised_samples = UnsupervisedSampler(
        learning_G, nodes=learning_nodes, length=length, number_of_walks=number_of_walks
    )
    batch_size = 50
    epochs = 10
    num_samples = [10, 5]
    generator = GraphSAGELinkGenerator(learning_G, batch_size, num_samples)
    train_gen = generator.flow(unsupervised_samples)
    layer_sizes = [128, 128]
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
    nodes = list(G.nodes())
    node_gen = GraphSAGENodeGenerator(G, batch_size, num_samples).flow(nodes)
    node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)

    return node_embeddings, nodes


def generate_node_embeddings_graphwave(G):
    sample_points = np.linspace(0, 50, 25).astype(np.float32)
    degree = 20
    scales = [2, 4]

    generator = GraphWaveGenerator(G, scales=scales, degree=degree)
    node_ids = G.nodes()
    embeddings_dataset = generator.flow(
        node_ids=node_ids, sample_points=sample_points, batch_size=1, repeat=False
    )

    embeddings = [x.numpy() for x in embeddings_dataset]
    embeddings = np.reshape(embeddings, (len(embeddings), len(embeddings[0][0])))
    embeddings_transformed = normalize_embeddings(embeddings)
    return embeddings_transformed, node_ids


def normalize_embeddings(embeddings):
    scaler = StandardScaler()
    scaler.fit(embeddings)
    embeddings_transformed = scaler.transform(embeddings)
    return embeddings_transformed

def generate_node_embeddings_deepgraphinfomax(G):
    fullbatch_generator = FullBatchNodeGenerator(G, sparse=False)
    gcn_model = GCN(layer_sizes=[128], activations=["relu"], generator=fullbatch_generator)

    corrupted_generator = CorruptedGenerator(fullbatch_generator)
    gen = corrupted_generator.flow(G.nodes())
    infomax = DeepGraphInfomax(gcn_model, corrupted_generator)
    x_in, x_out = infomax.in_out_tensors()

    model = Model(inputs=x_in, outputs=x_out)
    model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer=Adam(lr=1e-3))
    epochs = 100
    es = EarlyStopping(monitor="loss", min_delta=0, patience=20)
    history = model.fit(gen, epochs=epochs, verbose=0, callbacks=[es])
    plot_history(history)
    x_emb_in, x_emb_out = gcn_model.in_out_tensors()

    # for full batch models, squeeze out the batch dim (which is 1)
    x_out = tf.squeeze(x_emb_out, axis=0)
    emb_model = Model(inputs=x_emb_in, outputs=x_out)

    node_subjects = G.nodes()


    node_gen = fullbatch_generator.flow(node_subjects)
    embeddings = emb_model.predict(node_gen)
    return embeddings, node_subjects


def generate_node_embeddings_gcn(G):
    pass

def just_features_embeddings(G):
    embeddings = G.node_features()
    embeddings = normalize_embeddings(embeddings)
    return embeddings, G.nodes()

def combine_embeddings(embeddings_list, node_ids_list):
    embeddings = []
    node_ids = []
    dict_1 = dict(zip(node_ids_list[0], embeddings_list[0]))
    for i in range(1, len(embeddings_list)):
        for j in range(0, len(embeddings_list[i])):
            dict_1[node_ids_list[i][j]] = np.append(dict_1[node_ids_list[i][j]],embeddings_list[i][j])
    for key, value in dict_1.items():
        embeddings.append(value)
        node_ids.append(key)
    return embeddings, node_ids

def create_learning_G_from_true_matches_graphsage(G, true_matches):
    true_matches = [("u_"+t[0], "v_"+t[1]) for t in true_matches]
    learning_graph = G.copy() # keeping node attributes
    learning_graph = nx.create_empty_copy(learning_graph)
    learning_graph.add_edges_from(true_matches)
    learning_graph = learning_graph.subgraph(G.nodes())
    return StellarGraph.from_networkx(learning_graph, node_features="feature")

def generate_node_embeddings_custom(G):
    pass





