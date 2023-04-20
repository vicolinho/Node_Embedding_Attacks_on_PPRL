from itertools import chain

import networkx as nx
import numpy as np
from pandas import DataFrame
from stellargraph import StellarGraph

import attack.io_.inout
from attack.io_.argparser import argparser
from attack.io_ import inout, logs, import_data
from attack.blocking import blocking
from attack.preprocessing import preprocessing
from attack.sim_graph import sim_graph
from attack.evaluation_ import evaluation
from attack.features import node_features
from attack.node_matching_ import node_matching

import pandas as pd

from attack.node_embeddings import hyperparameter_tuning, embeddings
from classes.settings import Settings
from attack.preprocessing.preprocessing import BITARRAY, get_bigrams, QGRAMS
from attack.sim_graph.similarities import edges_df_from_blk_plain, edges_df_from_blk_bf_adjusted
from attack.sim_graph.analysis import false_negative_rate, get_num_hash_function

#DATA_PLAIN_FILE = "pprl_datasets/ncvoter-20140619-temporal-balanced-ratio-1to1-a.csv"
#DATA_ENCODED_FILE = "pprl_datasets/ncvoter-20140619-temporal-balanced-ratio-1to1-a_encoded_fn_ln.csv"
QGRAM_ATTRIBUTES = ['first_name', 'last_name']
BLK_ATTRIBUTES = ['first_name']#, 'last_name']
ENCODED_ATTR = 'base64_bf'
BF_LENGTH = 1024

def main():
    parser = argparser()
    settings = Settings(parser)
    print(settings.__dict__)
    if settings.mode == "graph_calc":
        combined_graph, true_matches = generate_graph(settings)
        if not settings.analysis:
            return
    else:
        combined_graph, true_matches = attack.io_.inout.load_graph_tp(graph_path=settings.pickle_file)
    calc_emb_analysis(combined_graph, settings, true_matches)


def calc_emb_analysis(combined_graph, settings, true_matches):
    embeddings_features = embeddings.just_features_embeddings(combined_graph, settings)
    matches_precision_output(embeddings_features, settings, true_matches, technique=embeddings.FEATURES)
    embedding_results_gen_graphsage = hyperparameter_tuning.embeddings_hyperparameter_graphsage_gen(combined_graph,
                                                                                                    hyperparameter_tuning.get_params_graphsage(settings.hp_config))
    embedding_results_gen_deepgraphinfomax = hyperparameter_tuning.embeddings_hyperparameter_deepgraphinfomax_gen(
        combined_graph, hyperparameter_tuning.get_params_deepgraphinfomax(settings.hp_config))
    embedding_results_gen_graphwave = hyperparameter_tuning.embeddings_hyperparameter_graphwave_gen(
        get_graph_for_original_graphwave(combined_graph, settings), hyperparameter_tuning.get_default_params_graphwave(settings.graphwave_libpath, settings.hp_config))
    #for embedding_results in chain(embedding_results_gen_graphsage, embedding_results_gen_deepgraphinfomax):
    for embedding_results in chain(embedding_results_gen_graphwave, embedding_results_gen_deepgraphinfomax, embedding_results_gen_graphsage):
        embedding_results = embedding_results.filter(embeddings_features.nodes)
        matches_precision_output(embedding_results, settings, true_matches, embedding_results.algo_settings.technique)
        weights = settings.weights
        for weight_selection in weights:
            merged_embeddings_results = embeddings_features.merge(embedding_results, weight1=weight_selection[0], weight2=weight_selection[1])
            matches_precision_output(merged_embeddings_results, settings, true_matches, embedding_results.algo_settings.technique, weights=weight_selection)

def get_graph_for_original_graphwave(graph, settings):
    if inout.graphwave_graph_exists(settings):
        G = inout.load_graph_for_graphwave_org(settings)
    else:
        G = StellarGraph.to_networkx(graph)
        inout.save_graph_for_graphwave_org(G, settings)
    return G

def generate_graph(settings):
    plain_data = import_data.import_data_plain(settings.plain_file, settings.record_count, QGRAM_ATTRIBUTES, BLK_ATTRIBUTES,
                                               ENCODED_ATTR, settings.padding)
    encoded_data = import_data.import_data_encoded(settings.encoded_file, settings.record_count, ENCODED_ATTR)
    true_matches = import_data.get_true_matches(plain_data[QGRAMS], encoded_data[ENCODED_ATTR])  # normal mode
    # true_matches = import_data.get_true_matches(plain_data[ENCODED_ATTR], encoded_data[ENCODED_ATTR]) # normal mode
    plain_data = plain_data.sample(frac=1 - settings.removed_plain_record_frac)
    encoded_data = encoded_data.sample(frac= 1 - settings.removed_encoded_record_frac)
    nodes_plain, edges_plain = create_sim_graph_plain(plain_data, ENCODED_ATTR, settings.threshold, BF_LENGTH, settings.lsh_count,
                                                      settings.lsh_size, num_of_hash_func=15, id='u')  # normal
    # nodes_plain, edges_plain = create_sim_graph_encoded(plain_data, ENCODED_ATTR, BF_LENGTH, lsh_count = 1, lsh_size = 0, num_of_hash_func=15, threshold = threshold, id = 'u')
    nodes_encoded, edges_encoded = create_sim_graph_encoded(encoded_data, ENCODED_ATTR, BF_LENGTH, settings.lsh_count, settings.lsh_size,
                                                            num_of_hash_func=15, threshold=settings.threshold, id='v')
    max_degree = max(len(nodes_plain), len(nodes_encoded)) - 1
    graph_plain = sim_graph.create_graph(nodes_plain, edges_plain, min_nodes=3, max_degree=max_degree,
                                         settings=settings)
    graph_encoded = sim_graph.create_graph(nodes_encoded, edges_encoded, min_nodes=3, max_degree=max_degree,
                                           settings=settings)
    combined_graph_nx = nx.compose(graph_plain, graph_encoded)
    combined_graph = StellarGraph.from_networkx(combined_graph_nx, node_features="feature")
    inout.save_graph_tp(combined_graph, true_matches, settings)
    return combined_graph, true_matches


def matches_precision_output(embeddings_results, settings, true_matches, technique, weights = [1.0]):
    matches = node_matching.matches_from_embeddings_combined_graph(embeddings_results, 'u', 'v', settings, weights)
    precision_list = []
    for top_pairs in settings.num_top_pairs:
        sub_matches = matches[:top_pairs]
        precision = evaluation.evalaute_top_pairs(sub_matches, true_matches, top_pairs)
        precision_list.append(precision)
        print(embeddings_results.info_string(), top_pairs, precision)
    if technique == embeddings.GRAPHWAVE:
        inout.output_results_csv_graphwave(precision_list, settings, len(matches), embeddings_results.algo_settings, weights)
    elif technique == embeddings.GRAPHSAGE:
        inout.output_results_csv_graphsage(precision_list, settings, len(matches), embeddings_results.algo_settings, weights)
    elif technique == embeddings.FEATURES:
        inout.output_results_csv_features(precision_list, settings, len(matches))
    elif technique == embeddings.DEEPGRAPHINFOMAX:
        inout.output_results_csv_deepgraphinfomax(precision_list, settings, len(matches), embeddings_results.algo_settings, weights)
    # attack.inout.output_result(embeddings_results.info_string(), precision_list, settings)



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
    blk_dicts_encoded = blocking.get_dict_dataframes_by_blocking_keys_encoded(encoded_data, encoded_attr, BITARRAY, bf_length, lsh_count, lsh_size)
    logs.log_blk_dist(blk_dicts_encoded)
    nodes = node_features.esti_qgram_count_encoded(encoded_data[BITARRAY], encoded_data[encoded_attr], bf_length, num_of_hash_func, id)
    edges = DataFrame()
    for blk_dict_encoded in blk_dicts_encoded:
        df_temp = edges_df_from_blk_bf_adjusted(blk_dict_encoded, threshold, encoded_attr,
                                                bf_length, num_of_hash_func, id)
        edges = pd.concat([edges, df_temp])
    return nodes, edges

def create_sim_graph_plain_old(plain_data, qgram_attributes, blk_attributes, blk_func, threshold, id):
    nodes = node_features.qgram_count_plain(plain_data[QGRAMS], id)
    blk_dicts_plain = blocking.get_dict_dataframes_by_blocking_keys_plain(plain_data, blk_attributes, blk_func)
    edges = edges_df_from_blk_plain(blk_dicts_plain, qgram_attributes, threshold, id)
    return nodes, edges

def create_sim_graph_plain(plain_data, encoded_attr, threshold, bf_length, lsh_count, lsh_size, num_of_hash_func, id):
    nodes = node_features.qgram_count_plain(plain_data[QGRAMS], id)
    blk_dicts_encoded = blocking.get_dict_dataframes_by_blocking_keys_encoded(plain_data, QGRAMS, QGRAMS, bf_length, lsh_count, lsh_size)
    logs.log_blk_dist(blk_dicts_encoded)
    edges = DataFrame()
    for blk_dict_encoded in blk_dicts_encoded:
        df_temp = edges_df_from_blk_plain(blk_dict_encoded, [], threshold, id) # qgram_attr isn't needed, not very clean code!
        edges = pd.concat([edges, df_temp])
    return nodes, edges


if __name__ == '__main__':
    import time
    start = time.time()
    main()
    end = time.time()
    print(end - start)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
