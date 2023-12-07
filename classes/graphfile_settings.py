class graphfile_settings:
    def __init__(self, graph_filename):
        """
        constructor from filename of graph pickle file
        :param graph_filename (str)

        pattern: c{rc}_t{t}(_comp{c})_r{rp}e{re}(_lshc{c})(_lshs{s})_{nf}{sca}.pkl
        inside {} are variables, all inside () is optional
        rc - record count, t - threshold, c - minimal connected component size, rp/re - fraction removed records plain/encoded,
        nf - node feature mode, sca - scaler of node features, c - lsh count blocking, s - lsh size blocking
        """
        file_without_extension = graph_filename[:graph_filename.rfind('.')]
        name_parts = file_without_extension.split("_")
        self.record_count = int(name_parts[0][1:])
        self.threshold = float(name_parts[1][1:])
        if "comp" in name_parts[2]:
            self.min_comp_size = int(name_parts[2][4:])
            name_parts.remove(name_parts[2])
        else:
            self.min_comp_size = 3
        self.set_removed_records(name_parts[2])
        self.lsh_count_blk = 1
        self.lsh_size_blk = 0 # both default values
        if len(name_parts) > 3:
            if "lshc" in name_parts[3]:
                self.lsh_count_blk = int(name_parts[3][4:])
                self.lsh_size_blk = int(name_parts[4][4:])
        if name_parts[-1] in ['minmax', 'std']:
            self.graph_scaled = name_parts[-1]
            name_parts.pop()
        if name_parts[-1] in ['all', 'egonet1', 'egonet2', 'fast', 'histo']:
            self.mode = "all" if "histo" == name_parts[-1] else name_parts[-1]
        else:
            self.mode = 'normal'


    def set_removed_records(self, removed_records_namepart):
        """
        extracts removing fraction from filename, needed to support legacy filename
        """
        e_pos = removed_records_namepart.find("e")
        if e_pos == -1:
            self.removed_records_plain = float(removed_records_namepart[1:])
            self.removed_records_encoded = 0.0
        else:
            self.removed_records_plain = float(removed_records_namepart[1:e_pos])
            self.removed_records_encoded = float(removed_records_namepart[e_pos+1:])
