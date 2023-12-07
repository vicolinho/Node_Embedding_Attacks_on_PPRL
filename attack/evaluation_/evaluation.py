def count_true_matches(predicted_matches, actual_matches):
    """
    counts true matches out of predicted ones
    :param predicted_matches: list of tuples with representations of a predicted matching pair
    :param actual_matches: list of tuples with representations of a actual matching pair
    :return: int: count of true matches
    """
    if len(predicted_matches) == 0:
        return float('nan')
    counter = 0
    for match in predicted_matches:
        if match in actual_matches:
            counter += 1
    return counter


def get_precisions(predicted_matches, actual_matches, num_top_pairs):
    """
    calculates precision values for different number of top predicted matches
    :param predicted_matches: list of tuples with representations of a predicted matching pair
    :param actual_matches: list of tuples with representations of a actual matching pair
    :param num_top_pairs: list of counts of how many top matches should be looked at
    :return: list: list with precisions with values at the same index as connected number of top pairs
    """
    precision_list = []
    for top_pairs in num_top_pairs:
        sub_matches = predicted_matches[:top_pairs]
        tp_count = count_true_matches(sub_matches, actual_matches)
        precision_list.append(tp_count / top_pairs)
    return precision_list