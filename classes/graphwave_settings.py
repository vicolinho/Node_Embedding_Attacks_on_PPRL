import attack.constants
from attack.node_embeddings import embeddings


class Graphwave_settings():

    def __init__(self, scales, sample_p_max_val, no_samples, order_approx=30):
        self.scales = scales
        self.sample_p_max_val = sample_p_max_val
        self.no_samples = no_samples
        self.order_approx = order_approx
        self.technique = attack.constants.GRAPHWAVE

    def __str__(self):
        return "{0} {1} (0,{2},{3}) order: {4}".format(
            self.technique, str(self.scales), str(self.sample_p_max_val),
                    str(self.no_samples), str(self.order_approx))

    def set_label_for_stellargraph_func(self):
        self.technique = attack.constants.GRAPHWAVE_OLD