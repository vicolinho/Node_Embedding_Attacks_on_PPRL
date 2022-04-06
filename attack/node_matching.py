import networkx as nx
import numpy as np
from ordered_set import OrderedSet
from pandas import DataFrame
from sklearn.metrics.pairwise import cosine_similarity
from queue import PriorityQueue

from stellargraph.globalvar import SOURCE, TARGET, WEIGHT

from attack import blocking

U = 'u'
V = 'v'

def bipartite_graph_to_matches(G, nodes1, nodes2, no_top_pairs):
    highest_pairs = PriorityQueue()
    u = [n for n in G.nodes if n[0] == U]
    try:
        matches = nx.bipartite.minimum_weight_full_matching(G, u)
    except ValueError:
        return []
    for node1, node2 in matches.items():
        if node1[0] == U:
            sim = G.get_edge_data(node1, node2)[WEIGHT]
            if highest_pairs.qsize() == no_top_pairs + 1:
                highest_pairs.get()
            nodeid1 = nodes1[int(node1[1:])]
            nodeid2 = nodes2[int(node2[1:])]
            highest_pairs.put((-sim, (nodeid1, nodeid2))) # reverse sim needed to biparitite match
    highest_pairs.get()
    matches = matches_from_priority_queue(highest_pairs)
    return matches

def embeddings_to_binary_lsh(embeddings1, embeddings2, dim):
    binary_lsh1 = np.zeros((len(embeddings1), dim)).astype(int)
    binary_lsh2 = np.zeros((len(embeddings2), dim)).astype(int)
    C = np.random.randn(len(embeddings1[0]), dim)
    for i in range(0, len(embeddings1)):
        binary_lsh1[i] = embeddings1[i].dot(C) >= 0
    for i in range(0, len(embeddings2)):
        binary_lsh2[i] = embeddings2[i].dot(C) >= 0
    return binary_lsh1, binary_lsh2

def create_dicts_lsh_embeddings(binary_lsh1, binary_lsh2, ids_list_hamming_lsh):
    lsh_dicts_embeddings = []
    for ids_hamming_lsh in ids_list_hamming_lsh:
        lsh_dict = {}
        for i in range(0, len(binary_lsh1)):
            add_id_to_lsh_dict(i, 0, ids_hamming_lsh, lsh_dict, binary_lsh1)
        for i in range(0, len(binary_lsh2)):
            add_id_to_lsh_dict(i, 1, ids_hamming_lsh, lsh_dict, binary_lsh2)
        lsh_dicts_embeddings.append(lsh_dict)
    return lsh_dicts_embeddings


def add_id_to_lsh_dict(i, j, ids_hamming_lsh, lsh_dict, lsh_emb):
    bitvector = blocking.lsh_blocking_key(lsh_emb[i], ids_hamming_lsh)
    #bitvector = [lsh_emb[i][j] for j in ids_hamming_lsh].tobytes()
    if not bitvector in lsh_dict:
        lsh_dict[bitvector] = [OrderedSet(), OrderedSet()]
    lsh_dict[bitvector][j].add(i)

def embeddings_to_bipartite_graph_lsh(embeddings1, embeddings2, threshold, hyperplane_count, lsh_count, lsh_size):
    binary_lsh1, binary_lsh2 = embeddings_to_binary_lsh(embeddings1, embeddings2, hyperplane_count)
    ids_list_hamming_lsh = blocking.choose_positions(lsh_count, lsh_size, hyperplane_count)
    lsh_dicts_embeddings = create_dicts_lsh_embeddings(binary_lsh1, binary_lsh2, ids_list_hamming_lsh)
    return embeddings_to_bipartite_graph_lsh_dicts(embeddings1, embeddings2, lsh_dicts_embeddings, threshold)



def embeddings_to_bipartite_graph_lsh_dicts(embeddings1, embeddings2, lsh_dicts_embeddings, threshold):
    # lsh_dicts_embeddings: list of dicts (key for comparison)
    # [{bitvector: [[u_ids,],[v_ids,]]}, ...]
    source, target, weight = [], [], []
    for lsh_dict_embeddings in lsh_dicts_embeddings:
        for value in lsh_dict_embeddings.values():
            temp_embeddings1 = [embeddings1[i] for i in value[0]]
            temp_embeddings2 = [embeddings2[i] for i in value[1]]
            if len(temp_embeddings1) == 0 or len(temp_embeddings2) == 0:
                continue
            cos_sims = cosine_similarity(temp_embeddings1, temp_embeddings2)
            for x in range(0, len(temp_embeddings1)):
                for y in range(0, len(temp_embeddings2)):
                    sim = cos_sims[x, y]
                    if sim >= threshold:
                        source.append(U + str(value[0][x]))
                        target.append(V + str(value[1][y]))
                        weight.append(-sim)  # needed to use minimum weight matching
    edges = DataFrame({SOURCE: source, TARGET: target, WEIGHT: weight})
    return nx.from_pandas_edgelist(edges, edge_attr=True)


def embeddings_to_bipartite_graph(embeddings1, embeddings2, threshold):
    source = []
    target = []
    weight = []
    cos_sims = cosine_similarity(embeddings1, embeddings2)
    for x in range(0, len(embeddings1)):
        for y in range(0, len(embeddings2)):
            sim = cos_sims[x,y]
            if sim >= threshold:
                source.append(U+str(x))
                target.append(V+str(y))
                weight.append(-sim) # needed to use minimum weight matching
    edges = DataFrame({SOURCE: source, TARGET: target, WEIGHT: weight})
    return nx.from_pandas_edgelist(edges, edge_attr=True)

def matches_from_embeddings_two_graphs(embeddings1, embeddings2, nodes1, nodes2, no_top_pairs, prefix_char=False,
                                       threshold=0.3):
    matches = bipartite_graph_to_matches(embeddings_to_bipartite_graph(embeddings1, embeddings2, threshold), nodes1, nodes2, no_top_pairs) # 719
    #matches = bipartite_graph_to_matches(embeddings_to_bipartite_graph_lsh(embeddings1, embeddings2, threshold, 20, 10, 8), nodes1, nodes2, no_top_pairs) 737
    if prefix_char:
        return remove_prefix_from_matches(matches)
    else:
        return matches

def matches_from_embeddings_combined_graph(embeddings, nodes, id1, id2, no_top_pairs, threshold = 0.3):
    embeddings1, embeddings2, nodes1, nodes2 = split_embeddings_by_nodes(embeddings, nodes, id1, id2)
    return matches_from_embeddings_two_graphs(embeddings1, embeddings2, nodes1, nodes2, no_top_pairs, True, threshold)


def split_embeddings_by_nodes(embeddings, nodes, prefix1, prefix2):
    embeddings1 = []
    embeddings2 = []
    nodes1 = []
    nodes2 = []
    for i in range(0, len(nodes)):
        if nodes[i][0] == prefix1:
            nodes1.append(nodes[i])
            embeddings1.append(embeddings[i])
        elif nodes[i][0] == prefix2:
            nodes2.append(nodes[i])
            embeddings2.append(embeddings[i])
    return embeddings1, embeddings2, nodes1, nodes2


def remove_prefix_from_matches(matches):
    adapted_matches = []
    for match in matches:
        adapted_match = (match[0][2:], match[1][2:])
        adapted_matches.append(adapted_match)
    return adapted_matches

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
