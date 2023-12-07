import pandas as pd


class Embedding_results:
    """
    encapsulates node embeddings with nodes and settings of node embedding technique
    """
    def __init__(self, embeddings, nodes, info_string_prefix, algo_settings = None):
        """
        :param embeddings (2d-array): node embeddings (or features)
        :param nodes (pd.Index(str)): node ids
        :param info_string_prefix (str): string info for technique
        :param algo_settings: encapsulation of parameters of node embedding technique (Graphsage_settings, Graphwave_settings, Deepgraphinfomax_settings)
        """
        if len(embeddings) > 5:
            self.embeddings = [embeddings]
        else:
            self.embeddings = embeddings
        self.nodes = nodes
        self.info_string_prefix = info_string_prefix
        self.algo_settings = algo_settings
        self.weights = []

    def merge(self, other, weight1, weight2):
        """
        merges two objects to one, useful for match selection when using both node features and node embeddings
        :param other (Embedding_results): object to merge with
        :param weight1 (float): weight for combined cosine similarity calculation
        :param weight2 (float): weight for combined cosine similarity calculation
        :return: Embedding_results: merged object
        """
        node_ids = []
        embeddings = [[], []]
        dict_1 = dict(zip(self.nodes, self.embeddings[0]))
        for i in range(0, len(other.nodes)):
            dict_1[other.nodes[i]] = [dict_1[other.nodes[i]], other.embeddings[0][i]]
        for key, value in dict_1.items():
            embeddings[0].append(value[0])
            embeddings[1].append(value[1])
            node_ids.append(key)
        if self.algo_settings == None:
            new_algo_settings = other.algo_settings
        elif other.algo_settings == None:
            new_algo_settings = self.algo_settings
        else:
            new_algo_settings = [self.algo_settings, other.algo_settings]
        merged = Embedding_results(embeddings, node_ids, self.info_string() + " + " + other.info_string(),
                                  new_algo_settings)
        merged.set_weights(weight1, weight2)
        return merged

    def set_weights(self, weight1, weight2):
        """
        sets weights for combined cosine similarity calculation and norms sum to 1
        :param weight1 (float)
        :param weight2 (float)
        """
        sum_weights = weight1 + weight2
        weight1 /= sum_weights
        weight2 /= sum_weights
        self.weights = [weight1, weight2]

    def filter(self, index):
        """
        creates new object just with information of selected node ids
        :param index (pd.Index(str)): node ids to be included
        :return: Embedding_results
        """
        df = pd.DataFrame(data=self.embeddings[0], index=self.nodes)
        df = df.loc[index]
        return Embedding_results(df.to_numpy(), df.index, self.info_string(), self.algo_settings)

    def info_string(self):
        """
        :return: str: description of used technique
        """
        if len(self.weights) == 0:
            return self.info_string_prefix
        else:
            return self.info_string_prefix + " w: " + str(self.weights)