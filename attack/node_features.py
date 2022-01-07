import networkx as nx
import numpy as np
import pandas as pd
from networkx import ego_graph, betweenness_centrality, degree_centrality
from numpy import Infinity
from stellargraph import StellarGraph

from attack import sim_graph
from attack.adjust_sims import compute_number_of_qgrams


def node_features_plain(series_qgrams, id):
    # series of DataFrame containing q-grams
    qgram_counts = series_qgrams.apply(len)
    qgrams_strings = series_qgrams.apply(sim_graph.adjust_node_id, args=(id))
    return pd.DataFrame({'qgrams':qgram_counts.to_numpy()}, index=qgrams_strings)

def node_features_encoded(series_bitarrays, series_encoded_attr ,bf_length, num_hash_f, id):
    # series of DataFrame containing q-grams
    bitarray_counts = series_bitarrays.apply(adjusted_number_of_qgrams, args=(bf_length, num_hash_f))
    id_strings = series_encoded_attr.apply(sim_graph.adjust_node_id, args=(id))
    return pd.DataFrame({'qgrams': bitarray_counts.to_numpy()}, index=id_strings)

def adjusted_number_of_qgrams(bitarray, bf_length, num_hash_f):
    number_of_bits = bitarray.count(1)
    return compute_number_of_qgrams(bf_length, num_hash_f, number_of_bits)

def add_node_features_vidange_networkx(G):
    # list of features (Vidange et al.)
    # NodeFreq
    # NodeLen
    # NodeDegr
    # EdgeMax
    # EdgeMin
    # EdgeAvr
    # EdgeStdDev
    # EgonetDegr
    # EgonetDens
    # BetwCentr
    # DegrCentr
    # OneHopHisto
    # TwoHopHisto
    G = nx.Graph(G) # if MultiGraph()
    betw_centr_dict = betweenness_centrality(G)
    degr_centr_dict = degree_centrality(G)
    for node_id in G.nodes:
        n = G.nodes[node_id]
        node_len = n['feature'][0]
        list_edge_weights = []
        for nbr, datadict in G.adj[node_id].items():
            list_edge_weights.append(datadict['weight'])
        node_degr = len(list_edge_weights)
        if node_degr == 0:
            n['feature'] = [node_len, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            continue
        edge_max = np.max(list_edge_weights)
        edge_min = np.min(list_edge_weights)
        edge_avg = np.mean(list_edge_weights)
        edge_std = np.std(list_edge_weights)
        egonet = ego_graph(G, node_id, radius=1, center=True, undirected=False, distance=None)
        egonet_node_count = len(egonet.nodes())
        egonet_degr = len(egonet.edges())
        egonet_dens = egonet_degr / (egonet_node_count / 2 * (egonet_node_count - 1))
        two_hop_egonet = ego_graph(G, node_id, radius=2, center=True, undirected=False, distance=None)
        one_hop_degrees = sorted([d for n, d in G.degree(egonet.nodes())], reverse=True)
        two_hop_degrees = sorted([d for n, d in G.degree(two_hop_egonet.nodes())], reverse=True)
        betw_centr = betw_centr_dict[node_id]
        degr_centr = degr_centr_dict[node_id]
        one_hop_histo = 0 # todo perhaps calculate max degree beforehand to determine size
        two_hop_histo = 0
        n['feature'] = [node_len, node_degr, edge_max, edge_min, edge_avg, edge_std, egonet_degr, egonet_dens, betw_centr, degr_centr, one_hop_histo, two_hop_histo]

    return G
