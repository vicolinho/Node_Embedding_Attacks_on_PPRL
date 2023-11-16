import time

import networkx as nx
import numpy as np
import pandas as pd
from pandas import DataFrame
from stellargraph import StellarGraph

from attack.blocking import blocking
from attack.constants import BITARRAY, NODE_COUNT, QGRAMS
from attack.features import node_features
from attack.io_ import logs
from attack.sim_graph.similarities import edges_df_from_blk_bf_adjusted, edges_df_from_blk_plain

STELLAR_GRAPH = 'stellargraph'

def adjust_node_id(name, prefix):
    return prefix + "_" + str(name)

def create_graph(nodes, edges, min_comp_size, settings):
    """
    creates networkx.Graph out of pd.Dataframe edge list and collects node features

    :param nodes:
    :param edges:
    :param min_comp_size:
    :param settings:
    :return:
    """
    G = nx.from_pandas_edgelist(edges, edge_attr=True)
    G = remove_small_comp_of_graph_nx(G, settings.min_comp_size)
    node_data = node_features.add_node_features_vidange_networkx(G, nodes, min_comp_size, settings)
    return G, node_data

def remove_small_comp_of_graph_nx(G, min_size):
    """
    :param G: networkx Graph (here a similarity graph)
    :param min_size (int): minimal size of connected component
    :return: networkx.Graph: new Graph without nodes of too small connected components
    """
    for comp in list(nx.connected_components(G)):
        if len(comp) < min_size:
            for node in comp:
                G.remove_node(node)
    return G

def remove_small_comp_of_graph_sg(G, min_size):
    """
    :param G: StellarGraph (here a similarity graph)
    :param min_size (int): minimal size of connected component
    :return: StellarGraph: new Graph without nodes of too small connected components
    """
    list_valid_nodes = np.array([])
    for comp in list(G.connected_components()):
        if len(comp) >= min_size:
            list_valid_nodes = np.append(list_valid_nodes, comp)
        else:
            break #it's valid because StellarGraph orders its connected components from big to small
    return G.subgraph(list_valid_nodes)


def calculate_sim_graph_data_encoded(encoded_data, settings, bf_length, num_of_hash_func, id):
    nodes = node_features.gph_indep_node_features_encoded(encoded_data[BITARRAY], encoded_data[settings.encoded_attr],
                                                          encoded_data[NODE_COUNT], bf_length, num_of_hash_func, id)
    edges = DataFrame()
    time_start = time.process_time()
    blk_dicts_encoded = blocking.get_dict_dataframes_by_blocking_keys(encoded_data, settings.encoded_attr, BITARRAY,
                                                                      bf_length, settings.lsh_count_blk, settings.lsh_size_blk)
    logs.log_blk_dist(blk_dicts_encoded)
    for blk_dict_encoded in blk_dicts_encoded:
        df_temp = edges_df_from_blk_bf_adjusted(blk_dict_encoded, settings.threshold, settings.encoded_attr,
                                                bf_length, num_of_hash_func, id)
        edges = pd.concat([edges, df_temp])
    time_end = time.process_time()
    print("Elapsed time (blocking), LSH (count, size):", time_end - time_start, settings.lsh_count_blk, settings.lsh_size_blk)
    return nodes, edges


def calculate_sim_graph_data_plain(plain_data, settings, bf_length, id):
    nodes = node_features.gph_indep_node_features_plain(plain_data[QGRAMS], plain_data[NODE_COUNT], id)
    time_start = time.process_time()
    blk_dicts_encoded = blocking.get_dict_dataframes_by_blocking_keys(plain_data, QGRAMS, QGRAMS, bf_length, settings.lsh_count_blk, settings.lsh_size_blk)
    logs.log_blk_dist(blk_dicts_encoded)
    edges = DataFrame()
    for blk_dict_encoded in blk_dicts_encoded:
        df_temp = edges_df_from_blk_plain(blk_dict_encoded, [], settings.threshold, id) # qgram_attr isn't needed, not very clean code!
        edges = pd.concat([edges, df_temp])
    time_end = time.process_time()
    print("Elapsed time (blocking), LSH (count, size):", time_end - time_start, settings.lsh_count_blk, settings.lsh_size_blk)
    return nodes, edges