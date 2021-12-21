from stellargraph import StellarGraph

from attack import import_data, ENCODED_ATTR, \
    create_sim_graph_encoded, BF_LENGTH, sim_graph, node_matching, evaluation

DATA_ENCODED_FILE = '../pprl_datasets/ncvoter-20140619-temporal-balanced-ratio-1to1-a_encoded_fn_ln.csv'

def graphwave_same_graph_encoded():
    embeddings_same_graph_encoded(sim_graph.generate_node_embeddings_graphwave)

def graphsage_same_graph_encoded():
    embeddings_same_graph_encoded(sim_graph.generate_node_embeddings_graphsage)

def node2vec_same_graph_encoded():
    embeddings_same_graph_encoded(sim_graph.generate_node_embeddings_node2vec)

def embeddings_same_graph_encoded(embedding_func):
    encoded_data = import_data.import_data_encoded(DATA_ENCODED_FILE, 100, ENCODED_ATTR)
    nodes_encoded, edges_encoded = create_sim_graph_encoded(encoded_data, ENCODED_ATTR, BF_LENGTH, lsh_count = 1, lsh_size = 0, num_of_hash_func=15, threshold = 0.4)
    graph_1 = StellarGraph(nodes_encoded, edges_encoded)
    graph_2 = StellarGraph(nodes_encoded, edges_encoded)
    embeddings_1, node_ids_1 = embedding_func(graph_1)
    embeddings_2, node_ids_2 = embedding_func(graph_2)
    matches = node_matching.matches_from_embeddings(embeddings_1, embeddings_2, node_ids_1, node_ids_2, 20)
    true_matches = [(id, id) for id in nodes_encoded.index.to_numpy()]
    precision = evaluation.evalaute_top_pairs(matches, true_matches)
    print(precision)

def main():
    graphwave_same_graph_encoded()
    # graphsage_same_graph_encoded()
    node2vec_same_graph_encoded()

if __name__ == '__main__':
    import time
    start = time.time()
    main()
    end = time.time()
    print(end - start)