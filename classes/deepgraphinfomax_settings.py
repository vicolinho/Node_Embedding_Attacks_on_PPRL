class Deepgraphinfomax_settings():

    def __init__(self, layers, activations):
        self.layers = layers
        self.activations = activations

    def __str__(self):
        return "{0} ({1}, {2})".format("deepgraphmax", str(self.layers), str(self.activations))
