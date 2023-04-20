from attack.node_embeddings import embeddings


class Deepgraphinfomax_settings():

    def __init__(self, layers, activations, epochs):
        self.layers = layers
        self.activations = activations
        self.technique = embeddings.DEEPGRAPHINFOMAX
        self.epochs = epochs

    def __str__(self):
        """
            needed for console output
        """
        return "{0} ({1}, {2}) ep: {3}".format("deepgraphinfomax", str(self.layers), str(self.activations), str(self.epochs))
