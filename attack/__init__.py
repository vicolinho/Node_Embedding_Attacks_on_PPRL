import networkx as nx
import numpy as np
from pandas import DataFrame
from stellargraph import StellarGraph
from stellargraph.datasets import datasets
from stellargraph.globalvar import SOURCE, TARGET

import attack.embeddings
from attack import blocking, preprocessing, sim_graph, node_matching, node_features, import_data, evaluation

import pandas as pd

from attack.preprocessing import BITARRAY, get_bigrams, QGRAMS
from attack.sim_graph import concat_edge_lists
from attack.similarities import edges_df_from_blk_plain, edges_df_from_blk_bf, edges_df_from_blk_bf_adjusted
from attack.analysis import false_negative_rate, get_num_hash_function

DATA_PLAIN_FILE = "pprl_datasets/ncvoter-20140619-temporal-balanced-ratio-1to1-a.csv"
DATA_ENCODED_FILE = "pprl_datasets/ncvoter-20140619-temporal-balanced-ratio-1to1-a_encoded_fn_ln.csv"
QGRAM_ATTRIBUTES = ['first_name', 'last_name']
BLK_ATTRIBUTES = ['first_name']#, 'last_name']
ENCODED_ATTR = 'base64_bf'
BF_LENGTH = 1024

def main():
    plain_data = import_data.import_data_plain(DATA_PLAIN_FILE, 1000, QGRAM_ATTRIBUTES, BLK_ATTRIBUTES)
    encoded_data = import_data.import_data_encoded(DATA_ENCODED_FILE, 1000, ENCODED_ATTR)
    true_matches = import_data.get_true_matches(plain_data[QGRAMS], encoded_data[ENCODED_ATTR])
    nodes_plain, edges_plain = create_sim_graph_plain(plain_data, QGRAM_ATTRIBUTES, BLK_ATTRIBUTES, blocking.no_blocking, 0.4, id = 'u')
    nodes_encoded, edges_encoded = create_sim_graph_encoded(encoded_data, ENCODED_ATTR, BF_LENGTH, lsh_count = 1, lsh_size = 0, num_of_hash_func=15, threshold = 0.4, id = 'v')
    graph_plain = StellarGraph(nodes_plain, edges_plain)
    graph_encoded = StellarGraph(nodes_encoded, edges_encoded)
    graph_plain = node_features.add_node_features_vidange_networkx(StellarGraph.to_networkx(graph_plain))
    graph_encoded = node_features.add_node_features_vidange_networkx(StellarGraph.to_networkx(graph_encoded))
    combined_graph = nx.compose(graph_plain, graph_encoded)
    graph_plain = StellarGraph.from_networkx(graph_plain, node_features="feature")
    graph_encoded = StellarGraph.from_networkx(graph_encoded, node_features="feature")
    combined_graph = StellarGraph.from_networkx(combined_graph, node_features="feature")
    embeddings_1, node_ids_1 = attack.embeddings.generate_node_embeddings_graphsage(graph_plain) # similiarities are way too high
    embeddings_2, node_ids_2 = attack.embeddings.generate_node_embeddings_graphsage(graph_encoded)
    matches = node_matching.matches_from_embeddings_two_graphs(embeddings_1, embeddings_2, node_ids_1, node_ids_2, 50, prefix_char=True)
    precision = evaluation.evalaute_top_pairs(matches, true_matches)
    print(precision)
    embeddings_comb, node_ids_comb = attack.embeddings.generate_node_embeddings_graphsage(combined_graph)
    matches = node_matching.matches_from_embeddings_combined_graph(embeddings_comb, node_ids_comb, 'u', 'v', 50)
    precision = evaluation.evalaute_top_pairs(matches, true_matches)
    print(precision)

def estimate_no_hash_func():
    encoded_data = pd.read_csv(DATA_ENCODED_FILE)
    #encoded_data = encoded_data.head(10000)
    encoded_data = preprocessing.preprocess_encoded_df(encoded_data, ENCODED_ATTR)
    plain_data = pd.read_csv(DATA_PLAIN_FILE, na_filter=False)
    #plain_data = plain_data.head(10000)
    cols = []
    for attribute in QGRAM_ATTRIBUTES:
        cols.append(plain_data[attribute])
    plain_data[QGRAMS] = list(map(get_bigrams, *cols))
    return get_num_hash_function(plain_data, encoded_data)

def play_around_with_lsh_parameters():
    encoded_data = pd.read_csv(DATA_ENCODED_FILE)
    encoded_data = encoded_data.head(3000)
    encoded_data = preprocessing.preprocess_encoded_df(encoded_data, ENCODED_ATTR)
    for i in np.arange(0.1, 1, 0.05):
        print(i, false_negative_rate(encoded_data, 70, 300, i))

def create_sim_graph_encoded(encoded_data, encoded_attr, bf_length, lsh_count, lsh_size, num_of_hash_func, threshold, id):
    blk_dicts_encoded = blocking.get_dict_dataframes_by_blocking_keys_encoded(encoded_data, bf_length, lsh_count, lsh_size)
    nodes = node_features.node_features_encoded(encoded_data[BITARRAY], encoded_data[encoded_attr], bf_length, num_of_hash_func, id)
    edges = DataFrame()
    for blk_dict_encoded in blk_dicts_encoded:
        df_temp = edges_df_from_blk_bf_adjusted(blk_dict_encoded, threshold, encoded_attr,
                                                bf_length, num_of_hash_func, id)
        edges = pd.concat([edges, df_temp])
    return nodes, edges

def create_sim_graph_plain(plain_data, qgram_attributes, blk_attributes, blk_func, threshold, id):
    nodes = node_features.node_features_plain(plain_data[QGRAMS], id)
    blk_dicts_plain = blocking.get_dict_dataframes_by_blocking_keys_plain(plain_data, blk_attributes, blk_func)
    edges = edges_df_from_blk_plain(blk_dicts_plain, qgram_attributes, threshold, id)
    return nodes, edges

if __name__ == '__main__':
    import time
    start = time.time()
    main()
    end = time.time()
    print(end - start)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
