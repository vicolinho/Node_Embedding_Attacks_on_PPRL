import math

import networkx as nx
import numpy as np
import pandas as pd
from networkx import ego_graph, betweenness_centrality, degree_centrality
from collections import Counter

from attack.sim_graph import sim_graph
from attack.sim_graph.adjust_sims import compute_number_of_qgrams


def qgram_count_plain(series_qgrams, id):
    # series of DataFrame containing q-grams
    qgram_counts = series_qgrams.apply(len)
    qgrams_strings = series_qgrams.apply(sim_graph.adjust_node_id, args=(id))
    return pd.DataFrame({'qgrams':qgram_counts.to_numpy()}, index=qgrams_strings)

def esti_qgram_count_encoded(series_bitarrays, series_encoded_attr, bf_length, num_hash_f, id):
    # series of DataFrame containing q-grams
    bitarray_counts = series_bitarrays.apply(adjusted_number_of_qgrams, args=(bf_length, num_hash_f))
    id_strings = series_encoded_attr.apply(sim_graph.adjust_node_id, args=(id))
    return pd.DataFrame({'qgrams': bitarray_counts.to_numpy()}, index=id_strings)

def adjusted_number_of_qgrams(bitarray, bf_length, num_hash_f):
    number_of_bits = bitarray.count(1)
    return compute_number_of_qgrams(bf_length, num_hash_f, number_of_bits)

def add_node_features_vidange_networkx(G, qgram_count_df, max_degree, settings):
    # List of features from Vidange et al. A Graph Matching Attack on PPRL
    # Histogram on log-scale (Heimann et al. REGAL: Representation Learning-Based Graph Alignment)
    G = nx.Graph(G) # if MultiGraph()
    if not settings.fast_mode:
        betw_centr_dict = betweenness_centrality(G)
    degr_centr_dict = degree_centrality(G)
    for node_id in G.nodes:
        n = G.nodes[node_id]
        node_len = qgram_count_df.loc[node_id]
        list_edge_weights = []
        for nbr, datadict in G.adj[node_id].items():
            list_edge_weights.append(datadict['weight'])
        node_degr = len(list_edge_weights)
        if node_degr == 0:
            G.remove_node(n)
            continue
        edge_max = np.max(list_edge_weights)
        edge_min = np.min(list_edge_weights)
        edge_avg = np.mean(list_edge_weights)
        edge_std = np.std(list_edge_weights)
        if not settings.fast_mode:
            egonet = ego_graph(G, node_id, radius=1, center=True, undirected=False, distance=None)
            egonet_node_count = len(egonet.nodes())
            egonet_degr = len(egonet.edges())
            egonet_dens = egonet_degr / (egonet_node_count / 2 * (egonet_node_count - 1))
            two_hop_egonet = ego_graph(G, node_id, radius=2, center=True, undirected=False, distance=None)
            one_hop_degrees = Counter([d for n, d in G.degree(egonet.nodes())])
            two_hop_degrees = Counter([d for n, d in G.degree(two_hop_egonet.nodes())])
            betw_centr = betw_centr_dict[node_id]
        degr_centr = degr_centr_dict[node_id]
        if settings.histo_features:
            one_hop_histo = counter_to_log_scale_histogram(one_hop_degrees, max_degree)
            two_hop_histo = counter_to_log_scale_histogram(two_hop_degrees, max_degree)
        else:
            one_hop_histo, two_hop_histo = [0], [0]
        if settings.fast_mode:
            n['feature'] = [node_len, node_degr, edge_max, edge_min, edge_avg, edge_std, degr_centr]
        else:
            n['feature'] = [node_len, node_degr, edge_max, edge_min, edge_avg, edge_std, egonet_degr, egonet_dens, betw_centr, degr_centr, *one_hop_histo, *two_hop_histo]

    return G

def counter_to_log_scale_histogram(counter, max_degree):
    histo = int(math.log(max_degree, 2)) * [0]
    for key, value in counter.items():
        histo[int(math.log(key, 2))] += value
    return histo

