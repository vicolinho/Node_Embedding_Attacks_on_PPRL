import attack.constants

class Deepgraphinfomax_settings():
    """
    stores some hyperparameter for DeepGraphInfomax
    """
    def __init__(self, layers, activations, epochs):
        """
        :param layers (list of int): neuron number of layers
        :param activations (list of str): activation function per layer
        :param epochs (int)
        """
        self.layers = layers
        self.activations = activations
        self.technique = attack.constants.DEEPGRAPHINFOMAX
        self.epochs = epochs

    def __str__(self):
        """
            needed for console output
        """
        return "{0} ({1}, {2}) ep: {3}".format("deepgraphinfomax", str(self.layers), str(self.activations), str(self.epochs))
