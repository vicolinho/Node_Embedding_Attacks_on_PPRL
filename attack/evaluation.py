from pathlib import Path

def evalaute_top_pairs(matches, true_matches):
    if len(matches) == 0:
        return float('nan')
    counter = 0
    for match in matches:
        if match in true_matches:
            counter += 1
    return counter / len(matches)

def output_result(technique, prec, output_path, record_count, threshold, removed_frac, histo_features, lsh_count, lsh_size):
    csv_header = "technique,prec\n"
    histo_suffix = '_histo' if histo_features else ''
    if lsh_size == 0:
        filename = "c{0}_t{1}_r{2}{3}.csv".format(record_count, threshold, removed_frac, histo_suffix)
    else:
        filename = "c{0}_t{1}_r{2}_lshc{3}_lshs{4}{5}.csv".format(record_count, threshold, removed_frac, lsh_count, lsh_size, histo_suffix)
    output_path = output_path + "/" if output_path[-1] != "/" else output_path
    full_path = output_path + filename
    header_needed = not Path(full_path).is_file()
    with open(full_path, 'a') as file:
        if header_needed:
            file.write(csv_header)
        file.write(technique + "," + str(prec) + "\n")

