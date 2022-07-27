import numpy as np
import pandas as pd


class Embedding_results:
    def __init__(self, embeddings, nodes, info_string_prefix):
        if len(embeddings) > 5:
            self.embeddings = [embeddings]
        else:
            self.embeddings = embeddings
        self.nodes = nodes
        self.info_string_prefix = info_string_prefix
        self.weights = []

    def merge(self, other, weight1, weight2):
        # merge only possible for two objects with one set of embeddings
        node_ids = []
        embeddings = [[], []]
        dict_1 = dict(zip(self.nodes, self.embeddings[0]))
        for i in range(0, len(other.nodes)):
            dict_1[other.nodes[i]] = [dict_1[other.nodes[i]], other.embeddings[0][i]]
           # dict_1[other.nodes[i]] = np.append(dict_1[other.nodes[i]], other.embeddings[i])
        for key, value in dict_1.items():
            embeddings[0].append(value[0])
            embeddings[1].append(value[1])
            node_ids.append(key)
        merged = Embedding_results(embeddings, node_ids, self.info_string() + " + " + other.info_string())
        merged.set_weights(weight1, weight2)
        return merged

    def set_weights(self, weight1, weight2):
        sum_weights = weight1 + weight2
        weight1 /= sum_weights
        weight2 /= sum_weights
        self.weights = [weight1, weight2]

    def filter(self, index):
        df = pd.DataFrame(data=self.embeddings[0], index=self.nodes)
        df = df.loc[index]
        return Embedding_results(df.to_numpy(), df.index, self.info_string())

    def info_string(self):
        if len(self.weights) == 0:
            return self.info_string_prefix
        else:
            return self.info_string_prefix + " w: " + str(self.weights)