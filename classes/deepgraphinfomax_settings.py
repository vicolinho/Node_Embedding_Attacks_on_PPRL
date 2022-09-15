from attack.node_embeddings import embeddings


class Deepgraphinfomax_settings():

    def __init__(self, layers, activations):
        self.layers = layers
        self.activations = activations
        self.technique = embeddings.DEEPGRAPHINFOMAX

    def __str__(self):
        return "{0} ({1}, {2})".format("deepgraphmax", str(self.layers), str(self.activations))
