def evalaute_top_pairs(matches, true_matches, no_top_pairs):
    if len(matches) == 0:
        return float('nan')
    counter = 0
    for match in matches:
        if match in true_matches:
            counter += 1
    return counter / no_top_pairs

