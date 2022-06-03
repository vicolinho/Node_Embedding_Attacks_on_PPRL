from itertools import chain

import networkx as nx
import numpy as np
from pandas import DataFrame
from stellargraph import StellarGraph

import attack.inout
from attack import embeddings, inout
from attack import blocking, preprocessing, sim_graph, node_matching, node_features, import_data, evaluation, \
    visualization

import pandas as pd
import argparse

from attack import hyperparameter_tuning
from classes.settings import Settings
from attack.preprocessing import BITARRAY, get_bigrams, QGRAMS
from attack.similarities import edges_df_from_blk_plain, edges_df_from_blk_bf_adjusted
from attack.analysis import false_negative_rate, get_num_hash_function

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
    lsh_count = int(parser.lsh_count)
    lsh_size = int(parser.lsh_size)
    if settings.mode == "graph_calc":
        removed_plain_record_frac = float(parser.remove_frac_plain)
        record_count = int(parser.record_count)
        threshold = float(parser.threshold)
        combined_graph, true_matches = generate_graph(lsh_count, lsh_size, parser, record_count, removed_plain_record_frac,
                                                  settings, threshold)
        if not settings.analysis:
            return
    else:
        combined_graph, true_matches = attack.inout.load_graph_tp(graph_path=settings.pickle_file)
    calc_emb_analysis(combined_graph, lsh_count, lsh_size, settings, true_matches)


def calc_emb_analysis(combined_graph, lsh_count, lsh_size, settings, true_matches):
    embeddings_features = embeddings.just_features_embeddings(combined_graph, settings)
    matches_precision_output(embeddings_features, lsh_size, lsh_count, settings, true_matches)
    embedding_results_gen_graphsage = hyperparameter_tuning.embeddings_hyperparameter_graphsage_gen(combined_graph,
                                                                                                 hyperparameter_tuning.get_default_params_graphsage())
    embedding_results_gen_deepgraphinfomax = hyperparameter_tuning.embeddings_hyperparameter_deepgraphinfomax_gen(
        combined_graph, hyperparameter_tuning.get_default_params_deepgraphinfomax())
    embedding_results_gen_graphwave = hyperparameter_tuning.embeddings_hyperparameter_graphwave_gen(
        combined_graph, hyperparameter_tuning.get_default_params_graphwave())
    #for embedding_results in chain(embedding_results_gen_graphsage, embedding_results_gen_deepgraphinfomax):
    for embedding_results in chain(embedding_results_gen_graphwave, embedding_results_gen_deepgraphinfomax, embedding_results_gen_graphsage):
        embedding_results = embedding_results.filter(embeddings_features.nodes)
        matches_precision_output(embedding_results, lsh_size, lsh_count, settings, true_matches)
        merged_embeddings_results = embeddings_features.merge(embedding_results)
        matches_precision_output(merged_embeddings_results, lsh_size, lsh_count, settings, true_matches)


def generate_graph(lsh_count, lsh_size, parser, record_count, removed_plain_record_frac, settings, threshold):
    plain_data = import_data.import_data_plain(parser.plain_file, record_count, QGRAM_ATTRIBUTES, BLK_ATTRIBUTES,
                                               ENCODED_ATTR)
    encoded_data = import_data.import_data_encoded(parser.encoded_file, record_count, ENCODED_ATTR)
    true_matches = import_data.get_true_matches(plain_data[QGRAMS], encoded_data[ENCODED_ATTR])  # normal mode
    # true_matches = import_data.get_true_matches(plain_data[ENCODED_ATTR], encoded_data[ENCODED_ATTR]) # normal mode
    plain_data = plain_data.sample(frac=1 - removed_plain_record_frac)
    nodes_plain, edges_plain = create_sim_graph_plain(plain_data, ENCODED_ATTR, threshold, BF_LENGTH, lsh_count,
                                                      lsh_size, num_of_hash_func=15, id='u')  # normal
    # nodes_plain, edges_plain = create_sim_graph_encoded(plain_data, ENCODED_ATTR, BF_LENGTH, lsh_count = 1, lsh_size = 0, num_of_hash_func=15, threshold = threshold, id = 'u')
    nodes_encoded, edges_encoded = create_sim_graph_encoded(encoded_data, ENCODED_ATTR, BF_LENGTH, lsh_count, lsh_size,
                                                            num_of_hash_func=15, threshold=threshold, id='v')
    max_degree = max(len(nodes_plain), len(nodes_encoded)) - 1
    graph_plain = sim_graph.create_graph(nodes_plain, edges_plain, min_nodes=3, max_degree=max_degree,
                                         histo_features=parser.histo_features)
    graph_encoded = sim_graph.create_graph(nodes_encoded, edges_encoded, min_nodes=3, max_degree=max_degree,
                                           histo_features=parser.histo_features)
    combined_graph_nx = nx.compose(graph_plain, graph_encoded)
    combined_graph = StellarGraph.from_networkx(combined_graph_nx, node_features="feature")
    inout.save_graph_tp(combined_graph, true_matches, settings)
    return combined_graph, true_matches


def matches_precision_output(merged_embeddings_results, lsh_size, lsh_count, settings, true_matches):
    matches = node_matching.matches_from_embeddings_combined_graph(merged_embeddings_results, 'u', 'v', settings)
    precision_list = []
    for top_pairs in settings.num_top_pairs:
        sub_matches = matches[:top_pairs]
        precision = evaluation.evalaute_top_pairs(sub_matches, true_matches)
        precision_list.append(precision)
        print(merged_embeddings_results.info_string, top_pairs, precision)
    attack.inout.output_result(merged_embeddings_results.info_string, precision_list, settings)


def print_precision_combined_embeddings(list_ids, embedding_func_names, embeddings_comb, node_ids_comb, results_path, record_count,
                                        removed_plain_record_frac, threshold, histo_features, true_matches, hyperplane_count, lsh_count, lsh_size):
    func_list, prec = prec_combined_embeddings(list_ids, embedding_func_names, embeddings_comb, node_ids_comb,
                                               true_matches, hyperplane_count, lsh_count, lsh_size)
    attack.inout.output_result(func_list, prec, results_path, record_count, threshold, removed_plain_record_frac, histo_features, lsh_count, lsh_size)


def prec_vis_embeddings(embeddings_comb, node_ids_comb, embedding_func_name, true_matches, settings):
    #visualization.vis(embeddings_comb, node_ids_comb, true_matches)
    matches = node_matching.matches_from_embeddings_combined_graph(embeddings_comb, node_ids_comb, 'u', 'v', settings)
    precision = evaluation.evalaute_top_pairs(matches, true_matches)
    print(embedding_func_name, precision)
    return embedding_func_name, precision


def prec_combined_embeddings(list_ids, embedding_func_names, embeddings_comb, node_ids_comb, true_matches, hyperplane_count = 1024, lsh_count = 1, lsh_size = 0):
    embeddings_comb_list = [embeddings_comb[i] for i in list_ids]
    ids_comb_list = [node_ids_comb[i] for i in list_ids]
    embeddings_func_list = ' '.join([embedding_func_names[i] for i in list_ids])

    emb, node_ids = embeddings.combine_embeddings(embeddings_comb_list, ids_comb_list)
    visualization.vis(emb, node_ids, true_matches)
    matches = node_matching.matches_from_embeddings_combined_graph(emb, node_ids, 'u', 'v', settings)
    precision = evaluation.evalaute_top_pairs(matches, true_matches)
    print(embeddings_func_list, precision)
    return embeddings_func_list, precision


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
    edges = DataFrame()
    for blk_dict_encoded in blk_dicts_encoded:
        df_temp = edges_df_from_blk_plain(blk_dict_encoded, [], threshold, id) # qgram_attr isn't needed, not very clean code!
        edges = pd.concat([edges, df_temp])
    return nodes, edges

def argparser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_full = subparsers.add_parser('graph_calc', help="save graph and calculate precision")
    parser_full.add_argument("--analysis", help='needed if analysis should be conducted', action='store_true')
    parser_full.add_argument("plain_file", help='path to plain dataset')
    parser_full.add_argument("encoded_file", help='path to encoded dataset')
    parser_full.add_argument("results_path", help='path to results output file')
    parser_full.add_argument("threshold", help='similarity threshold to be included in graph')
    parser_full.add_argument("--remove_frac_plain", help='fraction of plain records to be excluded')
    parser_full.add_argument("--record_count", help='restrict record count to be processed')
    parser_full.add_argument("--histo_features", help='adds histograms as features', action='store_true')
    parser_full.add_argument("--lsh_size", help='vector size for hamming lsh for indexing', default=0)
    parser_full.add_argument("--lsh_count", help='count of different lsh vectors for indexing', default=1)
    parser_full.add_argument("--min_edges", help='minimum edge count for a node to be matched', default=0)
    parser_full.add_argument("--graph_matching_tech", help='graph matching technique (shm, mwm, smm)', default='shm')


    parser_save_graph = subparsers.add_parser("graph_load", help='loading instead of calculating sim graph')
    parser_save_graph.add_argument("pickle_file", help="path to pickle file with graph and true matches")
    parser_save_graph.add_argument("results_path", help='path to results output file')
    parser_save_graph.add_argument("--lsh_size", help='vector size for hamming lsh for indexing', default=0)
    parser_save_graph.add_argument("--lsh_count", help='count of different lsh vectors for indexing', default=1)
    parser_save_graph.add_argument("--graph_matching_tech", help='graph matching technique (shm, mwm, smm)', default='shm')
    parser_save_graph.add_argument("--min_edges", help='minimum edge count for a node to be matched', default=0)
    #parser_save_graph.add_argument("--graphsage_settings_file", help='path to graphsage settings file for hyperparameter tuning')
    #parser_save_graph.add_argument("--deepgraphinfomax_settings_file", help='path to deepgraphinfomax settings file for hyperparameter tuning')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    import time
    start = time.time()
    main()
    end = time.time()
    print(end - start)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
