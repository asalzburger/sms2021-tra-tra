import numpy as np
import pandas as pd
import math


def get_str(x, n=2):
    """ returns a float as a string and cuts at the n-th decimal digit """
    x = truncate(x, n)
    x = 0.0 if x == 0.0 else x
    before, after = str(x).split('.', 1)
    return '.'.join([before, after[:n]])


def truncate(f, n):
    """ Truncates the digits of a floating point number. e.g.: f(0.8756343, 3) = 0.875 """
    return math.floor(f * 10 ** n) / 10 ** n


def find_bin(x, y, left_limit, lower_limit, width_bin_size, height_bin_size):
    """ Given a point in the Hough space, return the indices of the bin it falls into. """
    bin_x = int((x - left_limit - 1e-12) / width_bin_size)
    bin_y = int((y - lower_limit - 1e-12) / height_bin_size)
    return bin_x, bin_y


def update_tracks_per_bin(tracks_per_bin, track, bin_xs, bin_ys):
    """ Updates the information about which points (tracks in the original space) fall inside a given bin. """
    for bin_x, bin_y in zip(bin_xs, bin_ys):
        if (bin_x, bin_y) not in tracks_per_bin:
            tracks_per_bin[(bin_x, bin_y)] = [track]
        else:
            tracks_per_bin[(bin_x, bin_y)].append(track)


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


def pipeline(tracks, bin_size, limits, selection_hyperparameters, use_sin=False):
    """
    :param list[tuple[float, float]] tracks:
        A list with all the tracks (in the Hough space).

    :param tuple[float, float] bin_size:
        The size of each bin (from the binning process): (width, height).

    :param tuple[tuple[float, float], tuple[float, float]] limits:
        The range of selection for the values of the parameters (m, b).

    :param dict[str, int] selection_hyperparameters:
        A dictionary that provides us with the hyperprameters of the selection function (currently only minimum-hits-per-bin).

    :param bool use_sin:
        Boolean that determines whether to use the actual formula for q/p_T (with the sin)
        or to approximate `sin(phi_0 - phi)` with `phi_0 - phi`.


    :returns:
        A dictionary that maps an estimated track to the (truth) hits it convers.
    :rtype:
        dict[tuple[float, float], list[tuple[float, float]]]
    """
    A = 3e-4

    # range of search
    width_limits, height_limits = limits
    xs = np.arange(width_limits[0], width_limits[1], bin_size[0])
    accumulator = np.zeros((int((width_limits[1] - width_limits[0]) / bin_size[0]),
                            int((height_limits[1] - height_limits[0]) / bin_size[1])))
    flat_accumulator = np.ravel(accumulator)

    # vectorize the function of finding bins
    vect_find_bin = np.vectorize(find_bin)

    # book-keeping
    tracks_per_bin = {}
    for track in tracks:
        r, phi = track
        ys = np.sin(xs - phi) / (A * r) if use_sin else (xs - phi) / (A * r)

        # find in which bins the points belong to
        bin_xs, bin_ys = vect_find_bin(xs, ys, width_limits[0], height_limits[0], bin_size[0], bin_size[1])
        # flatten the indices in order to update the flat accumulator (which will automatically update the original accumulator)
        flat_indices = np.ravel_multi_index((bin_xs, bin_ys), accumulator.shape)
        flat_accumulator[flat_indices] += 1

        # update the book-keeping structure
        update_tracks_per_bin(tracks_per_bin, track, bin_xs, bin_ys)

    # get the indices of the bins that have at least the minimum required hits
    top_indices = bins_with_least_hits(tracks_per_bin, selection_hyperparameters['minimum-hits-per-bin'])

    # compute the coefficients (hence the tracks) from the location of the bin and book-keep which hits it contains
    est_tracks_to_hits = {}
    for bin_x, bin_y in top_indices:
        key_x = width_limits[0] + bin_x * bin_size[0]
        key_y = height_limits[0] + bin_y * bin_size[1]
        est_tracks_to_hits[(key_x, key_y)] = tracks_per_bin[(bin_x, bin_y)]

    return accumulator, est_tracks_to_hits

