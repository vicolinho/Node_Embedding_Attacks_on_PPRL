import networkx as nx
import numpy as np
from pandas import DataFrame
from sklearn.metrics.pairwise import cosine_similarity
from queue import PriorityQueue

from stellargraph.globalvar import SOURCE, TARGET, WEIGHT

U = 'u'
V = 'v'

def bipartite_graph_to_matches(G, nodes1, nodes2, no_top_pairs):
    highest_pairs = PriorityQueue()
    matches = nx.bipartite.minimum_weight_full_matching(G)
    for node1, node2 in matches.items():
        if node1[0] == U:
            sim = G.get_edge_data(node1, node2)[WEIGHT]
            if highest_pairs.qsize() == no_top_pairs + 1:
                highest_pairs.get()
            nodeid1 = nodes1[int(node1[1:])]
            nodeid2 = nodes2[int(node2[1:])]
            highest_pairs.put((sim, (nodeid1, nodeid2)))
    highest_pairs.get()
    matches = matches_from_priority_queue(highest_pairs)
    return matches


def embeddings_to_bipartite_graph(embeddings1, embeddings2): #todo: test it!
    source = []
    target = []
    weight = []
    embeddings1 = np.reshape(embeddings1, (len(embeddings1), len(embeddings1[0][0])))
    embeddings2 = np.reshape(embeddings2, (len(embeddings2), len(embeddings2[0][0])))
    cos_sims = cosine_similarity(embeddings1, embeddings2)
    for x in range(0, len(embeddings1)):
        for y in range(0, len(embeddings2)):
            source.append(U+str(x))
            target.append(V+str(y))
            weight.append(-cos_sims[x,y])
    edges = DataFrame({SOURCE: source, TARGET: target, WEIGHT: weight})
    return nx.from_pandas_edgelist(edges, edge_attr=True)

def matches_from_embeddings(embeddings1, embeddings2, nodes1, nodes2, no_top_pairs):
    return bipartite_graph_to_matches(embeddings_to_bipartite_graph(embeddings1, embeddings2), nodes1, nodes2, no_top_pairs)

def get_pqueue_pairs_ids_highest_sims(node_embeddings, ids1, ids2, no_top_pairs): # only needed graph1 <-> graph2
    highest_pairs = PriorityQueue()
    for i in ids1:
        for j in ids2:
            cos_sim = cosine_similarity(node_embeddings[i].reshape(1,-1), node_embeddings[j].reshape(1,-1))[0][0]
            if highest_pairs.qsize() == no_top_pairs + 1:
                highest_pairs.get()
            highest_pairs.put((cos_sim, (i, j)))
    highest_pairs.get()
    return highest_pairs

def get_pqueue_pairs_ids_highest_sims_two_graphs(node_embeddings_1, node_embeddings_2, no_top_pairs): # only needed graph1 <-> graph2
    highest_pairs = PriorityQueue()
    for i in range(0, len(node_embeddings_1)):
        for j in range(0, len(node_embeddings_2)):
            cos_sim = cosine_similarity(node_embeddings_1[i].reshape(1,-1), node_embeddings_2[j].reshape(1,-1))[0][0]
            if highest_pairs.qsize() == no_top_pairs + 1:
                highest_pairs.get()
            highest_pairs.put((cos_sim, (i, j)))
    highest_pairs.get()
    return highest_pairs

def matches_from_priority_queue(highest_pairs):
    matches = []
    while highest_pairs.qsize() != 0:
        pair = highest_pairs.get()
        matches.append((pair[1][0],pair[1][1]))
    return matches

def get_matching_node_ids_two_graphs(highest_pairs, node_ids_1, node_ids_2):
    matches = []
    while highest_pairs.qsize() != 0:
        pair = highest_pairs.get()
        matches.append((node_ids_1[pair[1][0]],node_ids_2[pair[1][1]]))
    return matches

def get_pairs_highest_sims_two_graphs(node_embeddings_1, node_embeddings_2, node_ids_1, node_ids_2, no_top_pairs):
    highest_pairs = get_pqueue_pairs_ids_highest_sims_two_graphs(node_embeddings_1, node_embeddings_2, no_top_pairs)
    return get_matching_node_ids_two_graphs(highest_pairs, node_ids_1, node_ids_2)


def get_matching_node_ids(highest_pairs, node_ids):
    while highest_pairs.qsize() != 0:
        pair = highest_pairs.get()
        print(node_ids[pair[1][0]],node_ids[pair[1][1]])


def get_pairs_highest_sims(node_embeddings, node_ids, no_top_pairs):
    ids1, ids2 = split_node_ids_into_two_groups(node_ids, '_')
    get_matching_node_ids(get_pqueue_pairs_ids_highest_sims(node_embeddings, ids1, ids2, no_top_pairs), node_ids)

def split_node_ids_into_two_groups(node_ids, split_char):
    ids1 = []
    ids2 = []
    for pos in range(0,len(node_ids)):
        if str(node_ids[pos])[-1] == split_char:
            ids2.append(pos)
        else:
            ids1.append(pos)
    return ids1, ids2
