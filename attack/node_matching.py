from sklearn.metrics.pairwise import cosine_similarity
from queue import PriorityQueue

def get_pqueue_pairs_ids_highest_sims(node_embeddings, no_top_pairs): # only needed graph1 <-> graph2
    highest_pairs = PriorityQueue()
    #half_size = int(len(node_embeddings) / 2)
    for i in range(0, len(node_embeddings)):
        for j in range(i+1, len(node_embeddings)):
            cos_sim = cosine_similarity(node_embeddings[i].reshape(1,-1), node_embeddings[j].reshape(1,-1))
            if highest_pairs.qsize() == no_top_pairs + 1:
                highest_pairs.get()
            highest_pairs.put((cos_sim, (i, j)))
    highest_pairs.get()
    return highest_pairs

def get_matching_node_ids(highest_pairs, node_ids):
    while highest_pairs:
        pair = highest_pairs.get()
        print(node_ids[pair[1][0]],node_ids[pair[1][1]])


def get_pairs_highest_sims(node_embeddings, node_ids, no_top_pairs):
    get_matching_node_ids(get_pqueue_pairs_ids_highest_sims(node_embeddings, no_top_pairs), node_ids)