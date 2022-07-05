class Settings:
    def __init__(self, parser):
        self.mode = parser.mode
        if self.mode == "graph_calc":
            self.record_count = int(parser.record_count)
            self.threshold = float(parser.threshold)
            self.removed_plain_record_frac = float(parser.remove_frac_plain)
            self.histo_features = bool(parser.histo_features)
            self.analysis = bool(parser.analysis)
            self.fast_mode = bool(parser.fast_mode)
        else:
            self.pickle_file = parser.pickle_file

        self.results_path = parser.results_path
        self.lsh_count = int(parser.lsh_count)
        self.lsh_size = int(parser.lsh_size)
        self.cos_sim_thold = float(0.3)
        self.hyperplane_count = int(1024)
        self.num_top_pairs = eval('[10, 50, 100, 500, 1000]')
        self.vidange_sim = bool(True)
        self.graph_matching_tech = parser.graph_matching_tech
        self.min_edges = int(parser.min_edges)
