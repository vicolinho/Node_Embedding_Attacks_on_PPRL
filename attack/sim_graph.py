import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from stellargraph import StellarGraph
from stellargraph.mapper import GraphSAGELinkGenerator, GraphWaveGenerator
from stellargraph.data import BiasedRandomWalk, UnsupervisedSampler
from stellargraph.globalvar import SOURCE, TARGET
from stellargraph.layer import GraphSAGE, link_classification
from tensorflow import keras


from attack import node_matching

STELLAR_GRAPH = 'stellargraph'



def read_both_graphs(graph_plain_file, graph_encoded_file, library = STELLAR_GRAPH):
    graph_plain = nx.read_gpickle(graph_plain_file)
    graph_encoded = nx.read_gpickle(graph_encoded_file)
    if library == STELLAR_GRAPH:
        graph_plain = StellarGraph.from_networkx(graph_plain)
        print(graph_plain.info())
        graph_encoded = StellarGraph.from_networkx(graph_encoded)
        print(graph_encoded.info())
    return graph_plain, graph_encoded

def duplicate_graph(graph_dataframe):
    # attributes first, second, weight
    return concat_edge_lists(graph_dataframe, graph_dataframe.copy())

def concat_edge_lists(edges_1, edges_2):
    # attributes first, second, weight
    len1 = len(edges_1)
    len2 = len(edges_2)
    for attr in [SOURCE, TARGET]:
        edges_2[attr] = edges_2[attr].astype(str) + '_'
        edges_2 = edges_2.set_index(pd.Index(range(len1, len1 + len2)))
    return pd.concat([edges_1, edges_2])

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
    node_matching.get_pairs_highest_sims(node_embeddings, node_ids, no_top_pairs=10)

    return model

def generate_node_embeddings_graphsage(G):
    # https://stellargraph.readthedocs.io/en/stable/demos/embeddings/graphsage-unsupervised-sampler-embeddings.html

    nodes = list(G.nodes())
    number_of_walks = 1
    length = 5
    unsupervised_samples = UnsupervisedSampler(
        G, nodes=nodes, length=length, number_of_walks=number_of_walks
    )
    batch_size = 50
    epochs = 4
    num_samples = [10, 5]
    generator = GraphSAGELinkGenerator(G, batch_size, num_samples)
    train_gen = generator.flow(unsupervised_samples)
    layer_sizes = [50, 50]
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

    return model

def generate_node_embeddings_graphwave(G):
    sample_points = np.linspace(0, 100, 50).astype(np.float32)
    degree = 20
    scales = [5, 10]

    generator = GraphWaveGenerator(G, scales=scales, degree=degree)
    node_ids = G.nodes()
    embeddings_dataset = generator.flow(
        node_ids=node_ids, sample_points=sample_points, batch_size=1, repeat=False
    )

    embeddings = [x.numpy() for x in embeddings_dataset]
    return embeddings, node_ids

