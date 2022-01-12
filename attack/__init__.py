import networkx as nx
import numpy as np
from pandas import DataFrame
from stellargraph import StellarGraph
from stellargraph.datasets import datasets
from stellargraph.globalvar import SOURCE, TARGET

import attack.embeddings
from attack import blocking, preprocessing, sim_graph, node_matching, node_features, import_data, evaluation, \
    visualization

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
    plain_data = import_data.import_data_plain(DATA_PLAIN_FILE, 2000, QGRAM_ATTRIBUTES, BLK_ATTRIBUTES)
    encoded_data = import_data.import_data_encoded(DATA_ENCODED_FILE, 2000, ENCODED_ATTR)
    true_matches = import_data.get_true_matches(plain_data[QGRAMS], encoded_data[ENCODED_ATTR])
    nodes_plain, edges_plain = create_sim_graph_plain(plain_data, QGRAM_ATTRIBUTES, BLK_ATTRIBUTES, blocking.no_blocking, 0.4, id = 'u')
    nodes_encoded, edges_encoded = create_sim_graph_encoded(encoded_data, ENCODED_ATTR, BF_LENGTH, lsh_count = 1, lsh_size = 0, num_of_hash_func=15, threshold = 0.4, id = 'v')
    graph_plain = StellarGraph(nodes_plain, edges_plain)
    graph_encoded = StellarGraph(nodes_encoded, edges_encoded)
    graph_plain = node_features.add_node_features_vidange_networkx(StellarGraph.to_networkx(graph_plain))
    graph_encoded = node_features.add_node_features_vidange_networkx(StellarGraph.to_networkx(graph_encoded))
    combined_graph = nx.compose(graph_plain, graph_encoded)
    combined_graph = sim_graph.remove_small_comp_of_graph(combined_graph, min_nodes=3)
    combined_graph = StellarGraph.from_networkx(combined_graph, node_features="feature")
    embedding_funcs = [attack.embeddings.just_features_embeddings,
                       attack.embeddings.generate_node_embeddings_graphsage,
                       attack.embeddings.generate_node_embeddings_graphwave,
                       attack.embeddings.generate_node_embeddings_node2vec
                       ]
    embedding_func_names = ['features', 'graphsage','graphwave', 'node2vec']
    embeddings_comb, node_ids_comb = [None] * len(embedding_funcs), [None] * len(embedding_funcs)
    for i in range(0, len(embedding_funcs)):
        embeddings_comb[i], node_ids_comb[i] = embedding_funcs[i](combined_graph)
        visualization.vis(embeddings_comb[i], node_ids_comb[i], true_matches)
        matches = node_matching.matches_from_embeddings_combined_graph(embeddings_comb[i], node_ids_comb[i], 'u', 'v', 50)
        precision = evaluation.evalaute_top_pairs(matches, true_matches)
        print(embedding_func_names[i], precision)
    for i in range(0, len(embedding_funcs)):
        for j in range(i+1, len(embedding_funcs)):
            emb, node_ids = attack.embeddings.combine_embeddings([embeddings_comb[i], embeddings_comb[j]], [node_ids_comb[i], node_ids_comb[j]])
            visualization.vis(emb, node_ids, true_matches)
            matches = node_matching.matches_from_embeddings_combined_graph(emb, node_ids, 'u',
                                                                           'v', 50)
            precision = evaluation.evalaute_top_pairs(matches, true_matches)
            print(embedding_func_names[i], embedding_func_names[j], precision)

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
