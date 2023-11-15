import math
from time import process_time

import networkx as nx
import numpy as np
import pandas as pd
from networkx import ego_graph, betweenness_centrality, degree_centrality
from collections import Counter
from sklearn import preprocessing

from attack.constants import QGRAMS_COUNT, NODE_COUNT
from attack.io_ import inout
from attack.sim_graph import sim_graph
from attack.sim_graph.adjust_sims import compute_number_of_qgrams


def gph_indep_node_features_plain(series_qgrams, series_node_count, id):
    # series of DataFrame containing q-grams
    qgram_counts = series_qgrams.apply(len)
    qgrams_strings = series_qgrams.apply(sim_graph.adjust_node_id, args=(id))
    return pd.DataFrame({QGRAMS_COUNT: qgram_counts.to_numpy(), NODE_COUNT: series_node_count.to_numpy()}, index=qgrams_strings)

def gph_indep_node_features_encoded(series_bitarrays, series_encoded_attr, series_node_count, bf_length, num_hash_f, id):
    # series of DataFrame containing q-grams
    bitarray_counts = series_bitarrays.apply(adjusted_number_of_qgrams, args=(bf_length, num_hash_f))
    id_strings = series_encoded_attr.apply(sim_graph.adjust_node_id, args=(id))
    return pd.DataFrame({QGRAMS_COUNT: bitarray_counts.to_numpy(), NODE_COUNT: series_node_count.to_numpy()}, index=id_strings)

def adjusted_number_of_qgrams(bitarray, bf_length, num_hash_f):
    number_of_bits = bitarray.count(1)
    return compute_number_of_qgrams(bf_length, num_hash_f, number_of_bits)

def add_node_features_vidange_networkx(G, nodes, max_degree, settings):
    # List of features from Vidange et al. A Graph Matching Attack on PPRL
    # Histogram on log-scale (Heimann et al. REGAL: Representation Learning-Based Graph Alignment)
    time_start = process_time()
    node_features = np.array([])
    G = nx.Graph(G) # if MultiGraph()
    connected_comps = (G.subgraph(c) for c in nx.connected_components(G))
    degr_centr_dict = dict()
    feature_set = str(settings.node_features)
    for connected_comp in connected_comps:
        degr_centr_dict.update(degree_centrality(connected_comp))
    if settings.node_features == 'all':
        betw_centr_dict = betweenness_centrality(G)
    for node_id in G.nodes:
        node_len = nodes.loc[node_id, QGRAMS_COUNT]
        node_count = nodes.loc[node_id, NODE_COUNT]
        list_edge_weights = []
        for nbr, datadict in G.adj[node_id].items():
            list_edge_weights.append(datadict['weight'])
        node_degr = len(list_edge_weights)
        if node_degr == 0:
            G.remove_node(G.nodes[node_id])
            continue
        edge_max = np.max(list_edge_weights)
        edge_min = np.min(list_edge_weights)
        edge_avg = np.mean(list_edge_weights)
        edge_std = np.std(list_edge_weights)
        degr_centr = degr_centr_dict[node_id]
        if settings.node_features == 'fast':
            features = [node_len, node_degr, edge_max, edge_min, edge_avg, edge_std, degr_centr]
        else:
            egonet = ego_graph(G, node_id, radius=1, center=True, undirected=False, distance=None)
            egonet_degr, egonet_dens = get_egonet_features(egonet)
            one_hop_histo = get_one_hop_histo(G, egonet, max_degree)
            if settings.node_features == 'egonet1':
                features = [node_len, node_degr, edge_max, edge_min, edge_avg, edge_std, egonet_degr,
                            egonet_dens, degr_centr, *one_hop_histo]
            else:
                two_hop_histo = get_two_hop_histo(G, max_degree, node_id)
                if settings.node_features == 'egonet2':
                    features = [node_len, node_degr, edge_max, edge_min, edge_avg, edge_std, egonet_degr,
                                egonet_dens, degr_centr, *one_hop_histo, *two_hop_histo]
                else:
                    betw_centr = betw_centr_dict[node_id]
                    features = [node_len, node_degr, edge_max, edge_min, edge_avg, edge_std, egonet_degr, egonet_dens,
                                betw_centr, degr_centr, *one_hop_histo, *two_hop_histo]

        if settings.node_count:
            feature_set += ", node_count"
            features.append(node_count)
        node_features = np.append(node_features, [features])
    node_features = np.reshape(node_features, (len(G.nodes), int(len(node_features) / len(G.nodes))))
    node_features = scale_node_features(node_features, settings)
    node_data = pd.DataFrame(node_features, index=G.nodes)
    time_end = process_time()
    elapsed_time = time_end - time_start
    print("Elapsed time (node features), Feature_Set:", elapsed_time, feature_set)
    inout.output_performance_node_features_results(elapsed_time, feature_set, settings)

    return node_data


def scale_node_features(arr, settings):
    if settings.graph_scaled == "standardscaler":
        scaler = preprocessing.StandardScaler()
        arr = scaler.fit_transform(arr)
    elif settings.graph_scaled == "minmaxscaler":
        scaler = preprocessing.MinMaxScaler()
        arr = scaler.fit_transform(arr)
    return arr


def get_one_hop_histo(G, egonet, max_degree):
    one_hop_degrees = Counter([d for n, d in G.degree(egonet.nodes())])
    one_hop_histo = counter_to_log_scale_histogram(one_hop_degrees, max_degree)
    return one_hop_histo


def get_two_hop_histo(G, max_degree, node_id):
    two_hop_egonet = ego_graph(G, node_id, radius=2, center=True, undirected=False, distance=None)
    two_hop_degrees = Counter([d for n, d in G.degree(two_hop_egonet.nodes())])
    two_hop_histo = counter_to_log_scale_histogram(two_hop_degrees, max_degree)
    return two_hop_histo


def get_egonet_features(egonet):
    egonet_node_count = len(egonet.nodes())
    egonet_degr = len(egonet.edges())
    egonet_dens = egonet_degr / (egonet_node_count / 2 * (egonet_node_count - 1))
    return egonet_degr, egonet_dens


def counter_to_log_scale_histogram(counter, max_degree):
    histo = int(math.log(max_degree, 2)) * [0]
    for key, value in counter.items():
        histo[int(math.log(key, 2))] += value
    return histo

