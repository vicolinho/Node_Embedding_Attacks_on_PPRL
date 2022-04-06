import networkx as nx
from stellargraph import StellarGraph

import node_features

STELLAR_GRAPH = 'stellargraph'

def adjust_node_id(name, prefix):
    return prefix + "_" + str(name)

def create_graph(nodes, edges, min_nodes, max_degree, histo_features):
    G = nx.from_pandas_edgelist(edges, edge_attr=True)
    G = remove_small_comp_of_graph(G, min_nodes)
    G = node_features.add_node_features_vidange_networkx(G, nodes, max_degree, histo_features)
    return G


def read_both_graphs(graph_plain_file, graph_encoded_file, library = STELLAR_GRAPH):
    graph_plain = nx.read_gpickle(graph_plain_file)
    graph_encoded = nx.read_gpickle(graph_encoded_file)
    if library == STELLAR_GRAPH:
        graph_plain = StellarGraph.from_networkx(graph_plain)
        print(graph_plain.info())
        graph_encoded = StellarGraph.from_networkx(graph_encoded)
        print(graph_encoded.info())
    return graph_plain, graph_encoded

def remove_small_comp_of_graph(G, min_nodes):
    for comp in list(nx.connected_components(G)):
        if len(comp) < min_nodes:
            for node in comp:
                G.remove_node(node)
    return G

