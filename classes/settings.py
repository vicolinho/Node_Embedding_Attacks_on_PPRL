class Settings:
    def __init__(self, parser): # todo datatypes
        self.mode = parser.mode
        if self.mode == "graph_calc":
            self.record_count = parser.record_count
            self.threshold = parser.threshold
            self.removed_plain_record_frac = parser.remove_frac_plain
            self.histo_features = parser.histo_features
            self.analysis = parser.analysis
        else:
            self.pickle_file = parser.pickle_file

        self.results_path = parser.results_path
        self.lsh_count = parser.lsh_count
        self.lsh_size = parser.lsh_size
        self.cos_sim_thold = 0.3
        self.hyperplane_count = 1024
        self.num_top_pairs = [10, 50, 100, 500, 1000]
        self.vidange_sim = True
        self.graph_matching_tech = parser.graph_matching_tech
        self.min_edges = parser.min_edges
