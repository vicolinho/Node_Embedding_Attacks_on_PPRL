import numpy as np


class Embedding_results:
    def __init__(self, embeddings, nodes, info_string):
        self.embeddings = embeddings
        self.nodes = nodes
        self.info_string = info_string

    def merge(self, other):
        embeddings = []
        node_ids = []
        dict_1 = dict(zip(self.nodes, self.embeddings))
        for i in range(0, len(other.embeddings)):
            dict_1[other.nodes[i]] = np.append(dict_1[other.nodes[i]], other.embeddings[i])
        for key, value in dict_1.items():
            embeddings.append(value)
            node_ids.append(key)
        merged = Embedding_results(embeddings, node_ids, self.info_string + " + " + other.info_string)
        return merged