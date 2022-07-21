class Graphwave_settings():

    def __init__(self, graphwave_libpath, scales, sample_p_max_val, no_samples, degree):
        self.graphwave_libpath = graphwave_libpath
        self.scales = scales
        self.sample_p_max_val = sample_p_max_val
        self.no_samples = no_samples
        self.degree = degree

    def __str__(self):
        return "{0} {1} (0,{2},{3}), deg:{4}".format(
            "graphwave_v2", str(self.scales), str(self.sample_p_max_val),
                    str(self.no_samples), str(self.degree))
