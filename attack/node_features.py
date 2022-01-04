import pandas as pd
from numpy import Infinity
from stellargraph import StellarGraph

from attack.adjust_sims import compute_number_of_qgrams


def node_features_plain(series_qgrams):
    # series of DataFrame containing q-grams
    qgram_counts = series_qgrams.apply(len)
    qgrams_strings = series_qgrams.apply(str)
    return pd.DataFrame({'qgrams':qgram_counts.to_numpy()}, index=qgrams_strings)

def node_features_encoded(series_bitarrays, series_encoded_attr ,bf_length, num_hash_f):
    # series of DataFrame containing q-grams
    bitarray_counts = series_bitarrays.apply(adjusted_number_of_qgrams, args=(bf_length, num_hash_f))
    id_strings = series_encoded_attr.apply(str)
    return pd.DataFrame({'qgrams': bitarray_counts.to_numpy()}, index=id_strings)

def adjusted_number_of_qgrams(bitarray, bf_length, num_hash_f):
    number_of_bits = bitarray.count(1)
    return compute_number_of_qgrams(bf_length, num_hash_f, number_of_bits)

def add_node_features_vidange(stellarG):
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
    nxG = StellarGraph.to_networkx(stellarG)
    for node_id in nxG.nodes:
        n = nxG.nodes[node_id]
        node_len = n['feature'][0]
        node_degr = 0
        edge_max = -Infinity
        edge_min = Infinity
        edge_avg = 0
        for nbr, datadict in nxG.adj[node_id].items():
            node_degr += 1
            edge_max = max(edge_max, datadict[0]['weight'])
            edge_min = min(edge_min, datadict[0]['weight'])
            edge_avg = edge_avg + (datadict[0]['weight'] - edge_avg) / node_degr
        n['feature'] = [node_len, node_degr, edge_max, edge_min, edge_avg]

    return StellarGraph.from_networkx(nxG, node_features="feature")
