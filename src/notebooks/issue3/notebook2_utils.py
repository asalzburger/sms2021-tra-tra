import numpy as np
import pandas as pd

from operator import itemgetter



def find_intersections(track_1, track_2):
    """ Finds the intersection point of the two given tracks. """
    m1, b1 = track_1
    m2, b2 = track_2
    intersection_x = - (b2 - b1) / (m2 - m1)
    intersection_y = m1 * intersection_x + b1
    return intersection_x, intersection_y


def compute_all_intersections(tracks):
    """ Returns a dictionary that maps a tuple of tracks IDs (their indices in the `tracks` list) to their intersection point """
    tracks_to_intersections = {}
    for idx_1, track_1 in enumerate(tracks):
        for idx_2, track_2 in enumerate(tracks[idx_1 + 1:]):
            intersection_x, intersection_y = find_intersections(track_1, track_2)
            tracks_to_intersections[(idx_1, idx_1 + 1 + idx_2)] = (intersection_x, intersection_y)
    return tracks_to_intersections


def get_limits(tracks):
    """ Returns the minimum and maximum values for each component (i.e. min/max(x-coordinates), min/max(y-coordinates)) """
    min_x = min(tracks, key=itemgetter(0))[0]
    max_x = max(tracks, key=itemgetter(0))[0]
    min_y = min(tracks, key=itemgetter(1))[1]
    max_y = max(tracks, key=itemgetter(1))[1]
    return (min_x, max_x), (min_y, max_y)


def find_bin(intersection_x, intersection_y, left_m_limit, left_b_limit, bin_size):
    """ Given a point in the Hough space, return the indices of the bin it falls into. """
    bin_x = int((intersection_x - left_m_limit - 1e-12) / bin_size[0])
    bin_y = int((intersection_y - left_b_limit - 1e-12) / bin_size[1])
    return bin_x, bin_y


def update_tracks_per_bin(tracks_per_bin, bin_x, bin_y, track_1, track_2):
    """ Updates the information about which points (tracks in the original space) fall inside a given bin. """
    if (bin_x, bin_y) not in tracks_per_bin:
        tracks_per_bin[(bin_x, bin_y)] = [track_1, track_2]
    else:
        tracks_per_bin[(bin_x, bin_y)].extend((track_1, track_2))


def remove_duplicate_tracks(tracks_per_bin):
    """ Removes duplicate tracks that might have been assigned to the same bin in the Hough space. """
    for _bin, tracks_in_it in tracks_per_bin.items():
        tracks_per_bin[_bin] = list(set(tracks_in_it))


def bins_with_least_intersections(accumulator, min_intersections):
    """ Returns a list of indices of the accumulator array that have a minimum value. """
    return np.transpose((accumulator >= min_intersections).nonzero())


def bins_with_least_hits(tracks_per_bin, min_hits):
    """ Returns a list of bins (tuple of ints) that have a minimum value. """
    return list(filter(lambda key: len(tracks_per_bin[key]) >= min_hits, tracks_per_bin))


def pipeline(tracks, intersections, bin_size, limits, selection_hyperparameters):
    """
    :param list[tuple[float, float]] tracks:
        A list with all the tracks (in the Hough space).

    :param dict[tuple[int, int], tuple[float, float]] intersections:
        A dictionary that maps: (index-of-track1, index-of-track2) -> their-intersection-point.

    :param tuple[float, float] bin_size:
        The size of each bin (from the binning process): (width, height).

    :param tuple[tuple[float, float], tuple[float, float]] limits:
        The range of selection for the values of the parameters (m, b).

    :param dict[str, int] selection_hyperparameters:
        A dictionary that provides us with the hyperprameters of the selection function (currently only minimum-hits-per-bin).

    
    :returns:
        A dictionary that maps an estimated track to the (truth) hits it convers.
    :rtype:
        dict[tuple[float, float], list[tuple[float, float]]]
    """
    # range of search for x_0 and y_0 values
    x_limits, y_limits = limits

    # dictionary that stores which tracks go through each bin
    tracks_per_bin = {}

    # loop for all the tracks in the Hough space
    for idx_1, track_1 in enumerate(tracks):
        for idx_2, track_2 in enumerate(tracks[idx_1 + 1:]):
            
            # retrieve the intersection point of the two tracks
            intersection_x, intersection_y = intersections[(idx_1, idx_1 + 1 + idx_2)]
            # get the bin it belongs to
            bin_x, bin_y = find_bin(intersection_x, intersection_y, x_limits[0], y_limits[0], bin_size)

            # add the tracks to the bin
            update_tracks_per_bin(tracks_per_bin, bin_x, bin_y, track_1, track_2)

    # remove duplicate tracks from inside the same bin
    remove_duplicate_tracks(tracks_per_bin)

    # get the indices of the bins that have at least the minimum required hits
    top_indices = bins_with_least_hits(tracks_per_bin, selection_hyperparameters['minimum-hits-per-bin'])

    # compute the coefficients (hence the tracks) from the location of the bin and book-keep which hits it contains
    est_tracks_to_hits = {}
    for bin_x, bin_y in top_indices:
        key_x = x_limits[0] + bin_x * bin_size[0]
        key_y = y_limits[0] + bin_y * bin_size[1]
        est_tracks_to_hits[(key_x, key_y)] = tracks_per_bin[(bin_x, bin_y)]

    return est_tracks_to_hits
