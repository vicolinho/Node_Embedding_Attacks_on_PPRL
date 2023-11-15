import attack.constants
from attack.node_embeddings import embeddings


class Graphsage_settings():

    def __init__(self, layers=None, num_samples=None, number_of_walks = 1, length = 5, batch_size = 50, epochs = 10):
        if num_samples is None:
            num_samples = [10, 5]
        if layers is None:
            layers = [128, 128]
        self.layers = layers
        self.number_of_walks = number_of_walks
        self.length = length
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_samples = num_samples
        self.technique = attack.constants.GRAPHSAGE

    def __str__(self):
        return "{0} (lay: {1}, |Samples|: {2}, |W|: {3}, W_len: {4}, |B|: {5}, Eps: {6})"\
            .format("graphsage", str(self.layers), str(self.num_samples), str(self.number_of_walks),
                    str(self.length), str(self.batch_size), str(self.epochs))
