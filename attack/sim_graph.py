import networkx as nx
import pandas as pd
from stellargraph import StellarGraph
from stellargraph.globalvar import SOURCE, TARGET

STELLAR_GRAPH = 'stellargraph'

def adjust_node_id(name, prefix):
    return prefix + "_" + str(name)

def read_both_graphs(graph_plain_file, graph_encoded_file, library = STELLAR_GRAPH):
    graph_plain = nx.read_gpickle(graph_plain_file)
    graph_encoded = nx.read_gpickle(graph_encoded_file)
    if library == STELLAR_GRAPH:
        graph_plain = StellarGraph.from_networkx(graph_plain)
        print(graph_plain.info())
        graph_encoded = StellarGraph.from_networkx(graph_encoded)
        print(graph_encoded.info())
    return graph_plain, graph_encoded

def concat_edge_lists(edges_1, edges_2):
    # attributes first, second, weight
    len1 = len(edges_1)
    len2 = len(edges_2)
    for attr in [SOURCE, TARGET]:
        edges_2[attr] = edges_2[attr].astype(str)
        edges_2 = edges_2.set_index(pd.Index(range(len1, len1 + len2)))
    return pd.concat([edges_1, edges_2])

