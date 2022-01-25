import networkx as nx
import numpy as np
from pandas import DataFrame
from stellargraph import StellarGraph

import embeddings
import blocking, preprocessing, sim_graph, node_matching, node_features, import_data, evaluation, \
    visualization

import pandas as pd
import argparse
from preprocessing import BITARRAY, get_bigrams, QGRAMS
from similarities import edges_df_from_blk_plain, edges_df_from_blk_bf, edges_df_from_blk_bf_adjusted
from analysis import false_negative_rate, get_num_hash_function

#DATA_PLAIN_FILE = "pprl_datasets/ncvoter-20140619-temporal-balanced-ratio-1to1-a.csv"
#DATA_ENCODED_FILE = "pprl_datasets/ncvoter-20140619-temporal-balanced-ratio-1to1-a_encoded_fn_ln.csv"
QGRAM_ATTRIBUTES = ['first_name', 'last_name']
BLK_ATTRIBUTES = ['first_name']#, 'last_name']
ENCODED_ATTR = 'base64_bf'
BF_LENGTH = 1024

def main():
    parser = argparser()
    removed_plain_record_frac = float(parser.remove_frac_plain)
    record_count = int(parser.record_count)
    threshold = float(parser.threshold)
    plain_data = import_data.import_data_plain(parser.plain_file, record_count, QGRAM_ATTRIBUTES, BLK_ATTRIBUTES)
    encoded_data = import_data.import_data_encoded(parser.encoded_file, record_count, ENCODED_ATTR)
    true_matches = import_data.get_true_matches(plain_data[QGRAMS], encoded_data[ENCODED_ATTR])
    plain_data = plain_data.sample(frac = 1 - removed_plain_record_frac)
    nodes_plain, edges_plain = create_sim_graph_plain(plain_data, QGRAM_ATTRIBUTES, BLK_ATTRIBUTES, blocking.no_blocking, threshold, id = 'u')
    nodes_encoded, edges_encoded = create_sim_graph_encoded(encoded_data, ENCODED_ATTR, BF_LENGTH, lsh_count = 1, lsh_size = 0, num_of_hash_func=15, threshold = threshold, id = 'v')
    graph_plain = sim_graph.create_graph(nodes_plain, edges_plain, min_nodes=3)
    graph_encoded = sim_graph.create_graph(nodes_encoded, edges_encoded, min_nodes=3)
    combined_graph_nx = nx.compose(graph_plain, graph_encoded)
    combined_graph = StellarGraph.from_networkx(combined_graph_nx, node_features="feature")
    embedding_funcs = [embeddings.just_features_embeddings,
                       embeddings.generate_node_embeddings_graphsage]
                       #embeddings.generate_node_embeddings_graphwave]
    embedding_func_names = ['features', 'graphsage',
      #  'graphwave',
        'graphsage_alt']
    embeddings_comb, node_ids_comb = [None] * len(embedding_func_names), [None] * len(embedding_func_names)
    for i in range(0, len(embedding_funcs)):
        embeddings_comb[i], node_ids_comb[i] = embedding_funcs[i](combined_graph)
        prec_vis_embeddings(embeddings_comb[i], node_ids_comb[i], embedding_func_names[i], true_matches)

    i = len(embedding_funcs)
    learning_G = embeddings.create_learning_G_from_true_matches_graphsage(combined_graph_nx, true_matches)
    embeddings_comb[i], node_ids_comb[i] = embeddings.generate_node_embeddings_graphsage(combined_graph, learning_G)
    prec_vis_embeddings(embeddings_comb[i], node_ids_comb[i], embedding_func_names[i], true_matches)

    prec_combined_embeddings([0,2], embedding_func_names, embeddings_comb, node_ids_comb, true_matches)
    #prec_combined_embeddings([0, 1, 2], embedding_func_names, embeddings_comb, node_ids_comb, true_matches)
    prec_combined_embeddings([0, 0, 2], embedding_func_names, embeddings_comb, node_ids_comb, true_matches)
    prec_combined_embeddings([0, 1, 0], embedding_func_names, embeddings_comb, node_ids_comb, true_matches)
    for i in range(0, len(embedding_funcs)):
        for j in range(i+1, len(embedding_funcs)):
            prec_combined_embeddings([i,j],embedding_func_names, embeddings_comb, node_ids_comb, true_matches)


def prec_vis_embeddings(embeddings_comb, node_ids_comb, embedding_func_name, true_matches):
    #embeddings_comb, node_ids_comb = embedding_func(combined_graph)
    visualization.vis(embeddings_comb, node_ids_comb, true_matches)
    matches = node_matching.matches_from_embeddings_combined_graph(embeddings_comb, node_ids_comb, 'u', 'v', 50,
                                                                   0.3)
    precision = evaluation.evalaute_top_pairs(matches, true_matches)
    print(embedding_func_name, precision)
    return embeddings_comb, node_ids_comb


def prec_combined_embeddings(list_ids, embedding_func_names, embeddings_comb, node_ids_comb, true_matches):
    embeddings_comb_list = [embeddings_comb[i] for i in list_ids]
    ids_comb_list = [node_ids_comb[i] for i in list_ids]
    embeddings_func_list = [embedding_func_names[i] for i in list_ids]

    emb, node_ids = embeddings.combine_embeddings(embeddings_comb_list, ids_comb_list)
    visualization.vis(emb, node_ids, true_matches)
    matches = node_matching.matches_from_embeddings_combined_graph(emb, node_ids, 'u', 'v', 50, 0.3)
    precision = evaluation.evalaute_top_pairs(matches, true_matches)
    print(*embeddings_func_list, precision)


def estimate_no_hash_func(encoded_file, plain_file, sample_size):
    encoded_data = pd.read_csv(encoded_file)
    encoded_data = encoded_data.head(sample_size)
    encoded_data = preprocessing.preprocess_encoded_df(encoded_data, ENCODED_ATTR)
    plain_data = pd.read_csv(plain_file, na_filter=False)
    plain_data = plain_data.head(sample_size)
    cols = []
    for attribute in QGRAM_ATTRIBUTES:
        cols.append(plain_data[attribute])
    plain_data[QGRAMS] = list(map(get_bigrams, *cols))
    return get_num_hash_function(plain_data, encoded_data)

def play_around_with_lsh_parameters(encoded_file):
    encoded_data = pd.read_csv(encoded_file)
    encoded_data = encoded_data.head(3000)
    encoded_data = preprocessing.preprocess_encoded_df(encoded_data, ENCODED_ATTR)
    for i in np.arange(0.1, 1, 0.05):
        print(i, false_negative_rate(encoded_data, 70, 300, i))

def create_sim_graph_encoded(encoded_data, encoded_attr, bf_length, lsh_count, lsh_size, num_of_hash_func, threshold, id):
    blk_dicts_encoded = blocking.get_dict_dataframes_by_blocking_keys_encoded(encoded_data, bf_length, lsh_count, lsh_size)
    nodes = node_features.esti_qgram_count_encoded(encoded_data[BITARRAY], encoded_data[encoded_attr], bf_length, num_of_hash_func, id)
    edges = DataFrame()
    for blk_dict_encoded in blk_dicts_encoded:
        df_temp = edges_df_from_blk_bf_adjusted(blk_dict_encoded, threshold, encoded_attr,
                                                bf_length, num_of_hash_func, id)
        edges = pd.concat([edges, df_temp])
    return nodes, edges

def create_sim_graph_plain(plain_data, qgram_attributes, blk_attributes, blk_func, threshold, id):
    nodes = node_features.qgram_count_plain(plain_data[QGRAMS], id)
    blk_dicts_plain = blocking.get_dict_dataframes_by_blocking_keys_plain(plain_data, blk_attributes, blk_func)
    edges = edges_df_from_blk_plain(blk_dicts_plain, qgram_attributes, threshold, id)
    return nodes, edges

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("plain_file", help='path to plain dataset')
    parser.add_argument("encoded_file", help='path to encoded dataset')
    parser.add_argument("threshold", help='similarity threshold to be included in graph')
    parser.add_argument("--remove_frac_plain", help='fraction')
    parser.add_argument("--record_count", help='restrict record count to be processed')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    import time
    start = time.time()
    main()
    end = time.time()
    print(end - start)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
