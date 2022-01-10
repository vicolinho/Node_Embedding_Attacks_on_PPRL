import networkx as nx
from stellargraph import StellarGraph
from stellargraph.datasets import datasets

import attack.embeddings
from attack import import_data, ENCODED_ATTR, \
    create_sim_graph_encoded, BF_LENGTH, sim_graph, node_matching, evaluation, QGRAM_ATTRIBUTES, BLK_ATTRIBUTES, \
    create_sim_graph_plain, blocking, node_features, embeddings, visualization

DATA_ENCODED_FILE = '../pprl_datasets/ncvoter-20140619-temporal-balanced-ratio-1to1-a_encoded_fn_ln.csv'
DATA_PLAIN_FILE = '../pprl_datasets/ncvoter-20140619-temporal-balanced-ratio-1to1-a.csv'

def graphwave_same_graph(graph_1, graph_2, true_matches):
    embeddings_two_graphs(attack.embeddings.generate_node_embeddings_graphwave, graph_1, graph_2, true_matches)

def graphsage_same_graph(graph_1, graph_2, true_matches):
    embeddings_two_graphs(attack.embeddings.generate_node_embeddings_graphsage, graph_1, graph_2, true_matches)

def node2vec_same_graph(graph_1, graph_2, true_matches):
    embeddings_two_graphs(attack.embeddings.generate_node_embeddings_node2vec, graph_1, graph_2, true_matches)

def graphinfomax_same_graph(graph_1, graph_2, true_matches):
    embeddings_two_graphs(attack.embeddings.generate_node_embeddings_deepgraphinfomax, graph_1, graph_2, true_matches)

def embeddings_two_graphs(embedding_func, graph_1, graph_2, true_matches):
    embeddings_1, node_ids_1 = embedding_func(graph_1)
    embeddings_2, node_ids_2 = embedding_func(graph_2)
    matches = node_matching.matches_from_embeddings_two_graphs(embeddings_1, embeddings_2, node_ids_1, node_ids_2, 20)
    precision = evaluation.evalaute_top_pairs(matches, true_matches)
    return precision

def embeddings_two_embeddings(embedding_func_1, embedding_func_2, combined_graph, true_matches):
    embeddings_1, node_ids_1 = embedding_func_1(combined_graph)
    embeddings_2, node_ids_2 = embedding_func_2(combined_graph)
    embeddings_comb, node_ids_comb = embeddings.combine_embeddings([embeddings_1, embeddings_2], [node_ids_1, node_ids_2])
    visualization.vis(embeddings_comb, node_ids_comb, true_matches)
    matches = node_matching.matches_from_embeddings_combined_graph(embeddings_comb, node_ids_comb, 'u', 'v', 20)
    precision = evaluation.evalaute_top_pairs(matches, true_matches)
    return precision


def embeddings_two_equal_graphs_in_one(embedding_func, combined_graph, true_matches):
    embeddings_comb, node_ids_comb = embedding_func(combined_graph)
    visualization.vis(embeddings_comb, node_ids_comb, true_matches)
    matches = node_matching.matches_from_embeddings_combined_graph(embeddings_comb, node_ids_comb, 'u', 'v', 20)
    precision = evaluation.evalaute_top_pairs(matches, true_matches)
    return precision

def get_graphs_and_matches_encoded():
    encoded_data = import_data.import_data_encoded(DATA_ENCODED_FILE, 200, ENCODED_ATTR)
    nodes_u, edges_u = create_sim_graph_encoded(encoded_data, ENCODED_ATTR, BF_LENGTH, lsh_count=1,
                                                            lsh_size=0, num_of_hash_func=15, threshold=0.4, id = 'u')
    nodes_v, edges_v = create_sim_graph_encoded(encoded_data, ENCODED_ATTR, BF_LENGTH, lsh_count=1,
                                                            lsh_size=0, num_of_hash_func=15, threshold=0.4, id='v')
    graph_1 = StellarGraph(nodes_u, edges_u)
    graph_2 = StellarGraph(nodes_v, edges_v)
    graph_1 = node_features.add_node_features_vidange_networkx(StellarGraph.to_networkx(graph_1))
    graph_2 = node_features.add_node_features_vidange_networkx(StellarGraph.to_networkx(graph_2))
    combined_graph = nx.compose(graph_1, graph_2)
    graph_1 = StellarGraph.from_networkx(graph_1, node_features="feature")
    graph_2 = StellarGraph.from_networkx(graph_2, node_features="feature")
    combined_graph = StellarGraph.from_networkx(combined_graph, node_features="feature")
    true_matches = [(id, 'v' + id[1:]) for id in nodes_u.index.to_numpy()]
    return graph_1, graph_2, combined_graph, true_matches

def get_graphs_and_matches_plain():
    plain_data = import_data.import_data_plain(DATA_PLAIN_FILE, 200, QGRAM_ATTRIBUTES, BLK_ATTRIBUTES)
    nodes_u, edges_u = create_sim_graph_plain(plain_data, QGRAM_ATTRIBUTES, BLK_ATTRIBUTES,
                                                      blocking.no_blocking, 0.4, id = 'u')
    nodes_v, edges_v = create_sim_graph_plain(plain_data, QGRAM_ATTRIBUTES, BLK_ATTRIBUTES,
                                                      blocking.no_blocking, 0.4, id='v')
    graph_1 = StellarGraph(nodes_u, edges_u)
    graph_2 = StellarGraph(nodes_v, edges_v)
    graph_1 = node_features.add_node_features_vidange_networkx(StellarGraph.to_networkx(graph_1))
    graph_2 = node_features.add_node_features_vidange_networkx(StellarGraph.to_networkx(graph_2))
    combined_graph = nx.compose(graph_1, graph_2)
    graph_1 = StellarGraph.from_networkx(graph_1, node_features="feature")
    graph_2 = StellarGraph.from_networkx(graph_2, node_features="feature")
    combined_graph = StellarGraph.from_networkx(combined_graph, node_features="feature")
    true_matches = [(id, 'v'+id[1:]) for id in nodes_u.index.to_numpy()]
    return graph_1, graph_2, combined_graph, true_matches

def get_graphs_and_matches_cora():
    dataset = datasets.Cora()
    G, node_subjects = dataset.load()
    true_matches = [(id, id) for id in node_subjects.index.to_numpy()]
    combined_graph = 0
    return G, G, combined_graph, true_matches


def main():
    get_graphs_and_matches_funcs = [get_graphs_and_matches_encoded, get_graphs_and_matches_plain, get_graphs_and_matches_cora]
    for func in get_graphs_and_matches_funcs[:2]:
        graph_1, graph_2, combined_graph, true_matches = func()
        embedding_funcs = [attack.embeddings.generate_node_embeddings_graphwave, attack.embeddings.generate_node_embeddings_graphsage,
                           attack.embeddings.generate_node_embeddings_node2vec]
        prec = embeddings_two_embeddings(attack.embeddings.generate_node_embeddings_graphwave, attack.embeddings.generate_node_embeddings_graphsage, combined_graph, true_matches)
        print("Precision one graph:", prec)
        for embedding_func in embedding_funcs[:2]:
            #prec = embeddings_two_graphs(embedding_func, graph_1, graph_2, true_matches)
            #print("Precision two graphs:",prec)
            prec2 = embeddings_two_equal_graphs_in_one(embedding_func, combined_graph, true_matches)
            print("Precision one graph:", prec2)

if __name__ == '__main__':
    import time
    start = time.time()
    main()
    end = time.time()
    print(end - start)