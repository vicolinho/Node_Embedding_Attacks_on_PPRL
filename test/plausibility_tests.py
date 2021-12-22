from stellargraph import StellarGraph
from stellargraph.datasets import datasets

from attack import import_data, ENCODED_ATTR, \
    create_sim_graph_encoded, BF_LENGTH, sim_graph, node_matching, evaluation, QGRAM_ATTRIBUTES, BLK_ATTRIBUTES, \
    create_sim_graph_plain, blocking

DATA_ENCODED_FILE = '../pprl_datasets/ncvoter-20140619-temporal-balanced-ratio-1to1-a_encoded_fn_ln.csv'
DATA_PLAIN_FILE = '../pprl_datasets/ncvoter-20140619-temporal-balanced-ratio-1to1-a.csv'

def graphwave_same_graph(graph_1, graph_2, true_matches):
    embeddings_same_graph(sim_graph.generate_node_embeddings_graphwave, graph_1, graph_2, true_matches)

def graphsage_same_graph(graph_1, graph_2, true_matches):
    embeddings_same_graph(sim_graph.generate_node_embeddings_graphsage, graph_1, graph_2, true_matches)

def node2vec_same_graph(graph_1, graph_2, true_matches):
    embeddings_same_graph(sim_graph.generate_node_embeddings_node2vec, graph_1, graph_2, true_matches)

def embeddings_same_graph(embedding_func, graph_1, graph_2, true_matches):
    embeddings_1, node_ids_1 = embedding_func(graph_1)
    embeddings_2, node_ids_2 = embedding_func(graph_2)
    matches = node_matching.matches_from_embeddings(embeddings_1, embeddings_2, node_ids_1, node_ids_2, 20)
    precision = evaluation.evalaute_top_pairs(matches, true_matches)
    print(precision)


def get_graphs_and_matches_encoded():
    encoded_data = import_data.import_data_encoded(DATA_ENCODED_FILE, 1000, ENCODED_ATTR)
    nodes_encoded, edges_encoded = create_sim_graph_encoded(encoded_data, ENCODED_ATTR, BF_LENGTH, lsh_count=1,
                                                            lsh_size=0, num_of_hash_func=15, threshold=0.2)
    graph_1 = StellarGraph(nodes_encoded, edges_encoded)
    graph_2 = StellarGraph(nodes_encoded, edges_encoded)
    true_matches = [(id, id) for id in nodes_encoded.index.to_numpy()]
    return graph_1, graph_2, true_matches

def get_graphs_and_matches_plain():
    plain_data = import_data.import_data_plain(DATA_PLAIN_FILE, 1000, QGRAM_ATTRIBUTES, BLK_ATTRIBUTES)
    nodes_plain, edges_plain = create_sim_graph_plain(plain_data, QGRAM_ATTRIBUTES, BLK_ATTRIBUTES,
                                                      blocking.no_blocking, 0.4)
    graph_1 = StellarGraph(nodes_plain, edges_plain)
    graph_2 = StellarGraph(nodes_plain, edges_plain)
    true_matches = [(id, id) for id in nodes_plain.index.to_numpy()]
    return graph_1, graph_2, true_matches

def get_graphs_and_matches_cora():
    dataset = datasets.Cora()
    G, node_subjects = dataset.load()
    true_matches = [(id, id) for id in node_subjects.index.to_numpy()]
    return G, G, true_matches


def main():
    for func in [get_graphs_and_matches_encoded, get_graphs_and_matches_plain, get_graphs_and_matches_cora]:
        graph_1, graph_2, true_matches = func()
        graphwave_same_graph(graph_1, graph_2, true_matches)
        graphsage_same_graph(graph_1, graph_2, true_matches)
        node2vec_same_graph(graph_1, graph_2, true_matches)

if __name__ == '__main__':
    import time
    start = time.time()
    main()
    end = time.time()
    print(end - start)