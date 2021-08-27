import numpy as np


def bins_with_least_hits(tracks_per_bin, min_hits):
    """ Returns a list of bins (tuple of ints) that have a minimum value. """
    return list(filter(lambda key: len(tracks_per_bin[key]) >= min_hits,
                       tracks_per_bin))


def least_square_distance(candidate_track, hits):
    """ Computes the mean of the square distances from an estimated track
        and the hits it "contains". """
    m, b = candidate_track
    _sum = 0.0
    for hit in hits:
        x, y = hit
        _sum += np.abs(m * x - y + b) / np.sqrt(np.square(m) + 1)
    return _sum / len(hits)
