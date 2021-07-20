import math
import numpy as np
import pandas as pd


def get_str(x, n=2):
    """ returns a float as a string and cuts at the n-th decimal digit """
    x = truncate(x, n)
    x = 0.0 if x == 0.0 else x
    before, after = str(x).split('.', 1)
    return '.'.join([before, after[:n]])


def truncate(f, n):
    """ Truncates the digits of a floating point number. e.g.: f(0.8756343, 3) = 0.875 """
    return math.floor(f * 10 ** n) / 10 ** n


def largest_indices(arr, n):
    """ Returns the n largest indices from a numpy array. """
    flat = arr.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, arr.shape)


def get_tracks(df):
    """ Returns a list with all the hits (tracks in the Hough space) and their row in the dataframe. """
    # book-keeping: store which particles form which track in the Hough space
    all_tracks = []
    hit_to_truth_df_row = {}
    for row, series in df.iterrows():
        r_i, z_i = series['r'], series['tz']
        hit = (-r_i, z_i)
        hit_to_truth_df_row[hit] = row
        # a hit becomes a track in the Hough space
        all_tracks.append(hit)
    
    return all_tracks, hit_to_truth_df_row


def find_intersections(track_1, track_2):
    """ Finds the intersection point of the two given tracks. """
    m1, b1 = track_1
    m2, b2 = track_2
    intersection_x = - (b2 - b1) / (m2 - m1)
    intersection_y = m1 * intersection_x + b1
    return intersection_x, intersection_y


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


def bins_with_least_hits(tracks_per_bin, min_hits):
    """ Returns a list of bins (tuple of ints) that have a minimum value. """
    return list(filter(lambda key: len(tracks_per_bin[key]) >= min_hits, tracks_per_bin))


def pipeline(df, bin_size, limits, selection_hyperparameters):
    """
    :param pd.DataFrame df:
        The dataframe with the hits information.

    :param tuple[float, float] bin_size:
        The size of each bin (from the binning process): (width, height).

    :param tuple[tuple[float, float], tuple[float, float]] limits:
        The range of selection for the values of the parameters (m, b).

    :param dict[str, int] selection_hyperparameters:
        A dictionary that provides us with the hyperprameters of the selection function (currently only minimum-hits-per-bin).
    """
    # get all the possible tracks in the Hough space and which row of the dataframe they correspond to
    all_tracks, _ = get_tracks(df)

    # range of search for slope (m) and intercept (b) values: bin size
    m_limits, b_limits = limits

    # dictionary that stores which tracks go through each bin
    tracks_per_bin = {}
    # loop for all the tracks in the Hough space
    for idx, track_1 in enumerate(all_tracks):

        # find intersections with all the other tracks
        for track_2 in all_tracks[idx+1:]:
            
            # find the intersection point of the two tracks
            intersection_x, intersection_y = find_intersections(track_1, track_2)

            # if the intersection point is outside of the boundary, ignore it
            if (not m_limits[0] <= intersection_x <= m_limits[1]) or not (b_limits[0] <= intersection_y <= b_limits[1]):
                continue

            # get the bin it belongs to
            bin_x, bin_y = find_bin(intersection_x, intersection_y, m_limits[0], b_limits[0], bin_size)

            # add the tracks to the bin
            update_tracks_per_bin(tracks_per_bin, bin_x, bin_y, track_1, track_2)

    # remove duplicate tracks from inside the same bin
    remove_duplicate_tracks(tracks_per_bin)

    # get the indices of the bins that have at least the minimum required hits
    top_indices = bins_with_least_hits(tracks_per_bin, selection_hyperparameters['minimum-hits-per-bin'])

    # compute the coefficients (hence the tracks) from the location of the bin and book-keep which hits it contains
    est_tracks_to_hits = {}
    for bin_x, bin_y in top_indices:
        key_m = m_limits[0] + bin_x * bin_size[0]
        key_b = b_limits[0] + bin_y * bin_size[1]
        est_tracks_to_hits[(key_m, key_b)] = tracks_per_bin[(bin_x, bin_y)]

    return est_tracks_to_hits