import networkx as nx

def convert_dict_to_graph(sim_dict):
    pass

def add_edges_from_sims(sim_array):
    list_edges = []
    return list_edges


def graph_test():
    G = nx.Graph()
    G.add_nodes_from(range(0,5))
    G.add_node('1qeefwf')
    G.add_node(10)
    G.add_edges_from([(1,2, {'weight': 0.5}),(3,5),(1,4)])
    return G