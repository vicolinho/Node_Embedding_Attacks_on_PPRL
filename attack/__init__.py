import networkx as nx
import numpy as np
from pandas import DataFrame
from stellargraph import StellarGraph
from stellargraph.globalvar import SOURCE, TARGET

from attack import blocking, preprocessing, sim_graph, node_matching, node_features

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
    nodes_plain, edges_plain = create_sim_graph_plain(DATA_PLAIN_FILE, QGRAM_ATTRIBUTES, BLK_ATTRIBUTES, blocking.no_blocking, 0.3, 1000)
    nodes_encoded, edges_encoded = create_sim_graph_encoded(DATA_ENCODED_FILE, ENCODED_ATTR, BF_LENGTH, lsh_count = 1, lsh_size = 0, threshold = 0.3, max_record_count=1000)
    #edges_plain = sim_graph.duplicate_graph(edges_plain)
    #edges_encoded = sim_graph.duplicate_graph(edges_encoded)
    #edges_plain = concat_edge_lists(edges_plain, edges_encoded)
    graph_plain = StellarGraph(edges=edges_plain, nodes=nodes_plain)

    print(graph_plain.info())
    graph_encoded = StellarGraph(edges=edges_encoded, nodes=nodes_encoded)
    print(graph_encoded.info())
    embeddings, node_ids = sim_graph.generate_node_embeddings_graphwave(graph_plain)
    node_matching.get_pairs_highest_sims(embeddings, node_ids, 100)

    G = nx.from_pandas_edgelist(edges_plain, edge_attr=True)
    print(G)
    sp = dict(nx.all_pairs_shortest_path(G))
    pass


def dummy_node_data(edges):
    nodes = pd.unique(pd.concat([edges[SOURCE], edges[TARGET]])).tolist()
    node_data = pd.DataFrame(
        len(nodes) * [1], index=pd.Index(data=nodes))
    return node_data


def measure_hash_fkt():
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

def create_sim_graph_encoded(file, encoded_attr, bf_length, lsh_count, lsh_size, threshold, max_record_count = -1):
    encoded_data = pd.read_csv(file)
    if max_record_count > 0:
        encoded_data = encoded_data.head(max_record_count)
    encoded_data = preprocessing.preprocess_encoded_df(encoded_data, encoded_attr)
    num_of_hash_func = 15 #todo implement a useful functionality independent of the used data sets
    blk_dicts_encoded = blocking.get_dict_dataframes_by_blocking_keys_encoded(encoded_data, bf_length, lsh_count, lsh_size)
    nodes = node_features.node_features_encoded(encoded_data[BITARRAY], encoded_data[encoded_attr], bf_length, num_of_hash_func)
    edges = DataFrame()
    for blk_dict_encoded in blk_dicts_encoded:
        df_temp = edges_df_from_blk_bf_adjusted(blk_dict_encoded, threshold, encoded_attr,
                                                bf_length, num_of_hash_func)
        edges = pd.concat([edges, df_temp])
    return nodes, edges

def create_sim_graph_plain(file, qgram_attributes, blk_attributes, blk_func, threshold, max_record_count = -1):
    plain_data = pd.read_csv(file, na_filter=False)
    if max_record_count > 0:
        plain_data = plain_data.head(max_record_count)
    plain_data = preprocessing.preprocess_plain_df(plain_data, qgram_attributes, blk_attributes)
    nodes = node_features.node_features_plain(plain_data[QGRAMS])
    blk_dicts_plain = blocking.get_dict_dataframes_by_blocking_keys_plain(plain_data, blk_attributes, blk_func)
    edges = edges_df_from_blk_plain(blk_dicts_plain, qgram_attributes, threshold)
    return nodes, edges

if __name__ == '__main__':
    import time
    start = time.time()
    main()
    end = time.time()
    print(end - start)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
