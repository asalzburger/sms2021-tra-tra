import numpy as np


def bins_with_least_intersections(accumulator, min_intersections):
    """ Returns a list of indices of the accumulator array that have a minimum value. """
    return np.transpose((accumulator >= min_intersections).nonzero())


def bins_with_least_hits(tracks_per_bin, min_hits):
    """ Returns a list of bins (tuple of ints) that have a minimum value. """
    return list(filter(lambda key: len(tracks_per_bin[key]) >= min_hits, tracks_per_bin))


def has_holes(candidate_track, tracks_per_bin):
    """ Determines whether a candidate track contains hits with holes in it. """
    pass


def least_square_distance(candidate_track, hits):
    """ Computes the mean of the square distances from an estimated track and the hits it "contains". """
    m, b = candidate_track
    _sum = 0.0
    for hit in hits:
        x, y = hit
        _sum += np.abs(m * x - y + b) / np.sqrt(np.square(m) + 1)
    return _sum / len(hits)


def select_bins(accumulator, bin_size, limits, min_hits, tracks_per_bin):
    """ Selects bins from an accumulator array. """
    
    # get the initial candidates by filtering out the tracks that contain a low number of intersections
    candidates = bins_with_least_intersections(accumulator, min_hits)

    # from those candidates filter out those that have holes
    candidates = []
    for idx, (bin_x, bin_y) in enumerate(candidates):
        m = -limits[0] + bin_x * bin_size[0]
        b = -limits[1] + bin_y * bin_size[1]
        candidate_track = (m, b)
        if has_holes(candidate_track, tracks_per_bin):
            del candidates[idx]

    # shared hits?

    # squared error
    square_errors = [least_square_distance(track, tracks_per_bin[track]) for track in candidates]
