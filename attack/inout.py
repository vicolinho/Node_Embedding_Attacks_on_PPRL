import os
import pickle
from pathlib import Path


def load_graph_tp(graph_path):
    with open(graph_path, 'rb') as input:
        tpl = pickle.load(input)
    return tpl[0], tpl[1]

def save_graph_tp(graph, true_matches, settings):
    subpath = "graphs"
    os.makedirs(subpath, exist_ok=True)
    filename = subpath + "/" + get_filename_template(settings) + ".pkl"
    tpl = (graph, true_matches)
    with open(filename, 'wb') as output:
        pickle.dump(tpl, output, pickle.HIGHEST_PROTOCOL)

def get_filename_template(settings):
    if settings.mode == "graph_calc":
        histo_suffix = '_histo' if settings.histo_features else ''
        if settings.lsh_size == 0:
            filename = "c{0}_t{1}_r{2}{3}".format(settings.record_count, settings.threshold,
                                                  settings.removed_plain_record_frac, histo_suffix)
        else:
            filename = "c{0}_t{1}_r{2}_lshc{3}_lshs{4}{5}".format(settings.record_count, settings.threshold,
                                                                  settings.removed_plain_record_frac,
                                                                  settings.lsh_count,
                                                                  settings.lsh_size, histo_suffix)
    else:
        filename = settings.pickle_file.split("/")[-1].split(".pkl")[0]
    return filename


def output_result(technique, prec, settings):#, output_path, record_count, threshold, removed_frac, histo_features, lsh_count, lsh_size):
    csv_header = "technique,prec\n"
    filename = get_filename_template(settings) + ".csv"
    output_path = settings.results_path + "/" if settings.results_path[-1] != "/" else settings.results_path
    full_path = output_path + filename
    header_needed = not Path(full_path).is_file()
    with open(full_path, 'a') as file:
        if header_needed:
            file.write(csv_header)
        file.write(technique + "," + str(prec) + "\n")