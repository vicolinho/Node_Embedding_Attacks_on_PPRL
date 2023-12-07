import attack.constants

class Graphwave_settings():
    """
    encapsulates some settings for GraphWave hyperparameter tuning
    """

    def __init__(self, scales, sample_p_max_val, no_samples, order_approx=30):
        """
        :param scales (list of int/str): scaling parameter
        :param sample_p_max_val: max value for sample points
        :param no_samples: number of sample points
        :param order_approx (int): order of approximation of characteristic function
        """
        self.scales = scales
        self.sample_p_max_val = sample_p_max_val
        self.no_samples = no_samples
        self.order_approx = order_approx
        self.technique = attack.constants.GRAPHWAVE

    def __str__(self):
        """
        used for console output
        """
        return "{0} {1} (0,{2},{3}) order: {4}".format(
            self.technique, str(self.scales), str(self.sample_p_max_val),
                    str(self.no_samples), str(self.order_approx))

    def set_label_for_stellargraph_func(self):
        """
        change technique label if StellarGraph is used for GraphWave
        """
        self.technique = attack.constants.GRAPHWAVE_OLD