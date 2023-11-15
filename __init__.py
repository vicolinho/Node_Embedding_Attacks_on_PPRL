from itertools import chain

import networkx as nx
import pandas as pd
from stellargraph import StellarGraph

import attack.constants
import attack.io_.inout
from attack.constants import QGRAMS
from attack.evaluation_.evaluation import get_precisions
from attack.io_.argparser import argparser
from attack.io_ import inout, import_data
from attack.sim_graph import sim_graph
from attack.node_matching_ import node_matching

from attack.node_embeddings import hyperparameter_tuning, embeddings
from attack.sim_graph.sim_graph import calculate_sim_graph_data_encoded, calculate_sim_graph_data_plain
from classes.settings import Settings
from attack.preprocessing.preprocessing import preprocess_dfs

def main():
    parser = argparser()
    settings = Settings(parser)
    print(settings.__dict__)
    if settings.mode == 'graph_load':
        combined_graph, actual_matches = attack.io_.inout.load_graph_tp(graph_path=settings.pickle_file)
        if settings.min_comp_size:
            combined_graph = sim_graph.remove_small_comp_of_graph(combined_graph, settings.min_comp_size)
    else:
        combined_graph, actual_matches = generate_graph(settings)
        if settings.mode == "graph_save":
            return
    calc_emb_analysis(combined_graph, settings, actual_matches)


def calc_emb_analysis(combined_graph, settings, true_matches):
    embeddings_features = embeddings.just_features_embeddings(combined_graph, settings)
    matches_precision_output(embeddings_features, settings, true_matches, technique=attack.constants.FEATURES)
    embedding_results_gen_graphsage = hyperparameter_tuning.embeddings_hyperparameter_graphsage_gen(combined_graph,
                                                                                                    hyperparameter_tuning.get_params_graphsage(settings.hp_config), settings.scaler)
    embedding_results_gen_deepgraphinfomax = hyperparameter_tuning.embeddings_hyperparameter_deepgraphinfomax_gen(
        combined_graph, hyperparameter_tuning.get_params_deepgraphinfomax(settings.hp_config), settings.scaler)
    embedding_results_gen_graphwave = hyperparameter_tuning.embeddings_hyperparameter_graphwave_gen(combined_graph,
                                                                                                    hyperparameter_tuning.get_default_params_graphwave(settings.graphwave_external_lib, settings.hp_config),
                                                                                                    settings)

    for embedding_results in chain(embedding_results_gen_graphwave, embedding_results_gen_deepgraphinfomax, embedding_results_gen_graphsage):
        embedding_results = embedding_results.filter(embeddings_features.nodes)
        matches_precision_output(embedding_results, settings, true_matches, embedding_results.algo_settings.technique)
        weights = settings.weights
        for weight_selection in weights:
            merged_embeddings_results = embeddings_features.merge(embedding_results, weight1=weight_selection[0], weight2=weight_selection[1])
            matches_precision_output(merged_embeddings_results, settings, true_matches,
                                     embedding_results.algo_settings.technique, weights=weight_selection)


def generate_graph(settings):
    plain_data, encoded_data, actual_matches, no_hash_func, bf_length = get_record_data_df(settings)
    nodes_plain, edges_plain = calculate_sim_graph_data_plain(plain_data, settings, bf_length, id='u')  # normal
    nodes_encoded, edges_encoded = calculate_sim_graph_data_encoded(encoded_data, settings, bf_length, no_hash_func, id='v')
    max_degree = max(len(nodes_plain), len(nodes_encoded)) - 1
    graph_plain, node_data_plain = sim_graph.create_graph(nodes_plain, edges_plain, max_degree, settings)
    graph_encoded, node_data_encoded = sim_graph.create_graph(nodes_encoded, edges_encoded, max_degree, settings)
    combined_graph_nx = nx.compose(graph_plain, graph_encoded)
    node_data = node_data_plain.append(node_data_encoded, ignore_index = False)
    combined_graph = StellarGraph.from_networkx(combined_graph_nx, node_features=node_data)
    inout.save_graph_tp(combined_graph, actual_matches, settings)
    return combined_graph, actual_matches

def get_record_data_df(settings):
    plain_data = import_data.import_data(settings.plain_file, settings.record_count, settings.removed_plain_record_frac, random_state=3)
    encoded_data = import_data.import_data(settings.encoded_file, settings.record_count, settings.removed_encoded_record_frac, random_state=5)
    plain_data, encoded_data, no_hash_func, bf_length = preprocess_dfs(encoded_data, plain_data, settings)
    actual_matches = list(pd.merge(plain_data[QGRAMS].apply(str), encoded_data[settings.encoded_attr], left_index=True, right_index=True).itertuples(
        index=False, name=None))
    return plain_data, encoded_data, actual_matches, no_hash_func, bf_length


def matches_precision_output(embeddings_results, settings, actual_matches, technique, weights=[1.0]):
    predicted_matches = node_matching.matches_from_embeddings_combined_graph(embeddings_results, 'u', 'v', settings, weights)
    precision_list = get_precisions(predicted_matches, actual_matches, settings.num_top_pairs)
    print(embeddings_results.info_string(), settings.num_top_pairs, precision_list)
    if technique in [attack.constants.GRAPHWAVE, attack.constants.GRAPHWAVE_OLD]:
        inout.output_results_csv_graphwave(precision_list, settings, len(predicted_matches), embeddings_results.algo_settings, weights)
    elif technique == attack.constants.GRAPHSAGE:
        inout.output_results_csv_graphsage(precision_list, settings, len(predicted_matches), embeddings_results.algo_settings, weights)
    elif technique == attack.constants.FEATURES:
        inout.output_results_csv_features(precision_list, settings, len(predicted_matches))
    elif technique == attack.constants.DEEPGRAPHINFOMAX:
        inout.output_results_csv_deepgraphinfomax(precision_list, settings, len(predicted_matches), embeddings_results.algo_settings, weights)


if __name__ == '__main__':
    import time
    start = time.time()
    main()
    end = time.time()
    print(end - start)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
