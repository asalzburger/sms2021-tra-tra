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


def pipeline(df, bin_size, limits, num_tracks_to_estimate):
    """
    :param pd.DataFrame df:
        The dataframe with the hits information.

    :param tuple[float, float] bin_size:
        The size of each bin (from the binning process): (width, height).

    :param tuple[tuple[float, float], tuple[float, float]] limits:
        The range of selection for the values of the parameters (m, b).

    :param int num_tracks_to_estimate:
        Number of tracks to sample from the accumulator.
        Later will remove this, as selection will occur with other methods.
    """
    # get all the possible tracks in the Hough space and which row of the dataframe they correspond to
    all_tracks, hit_to_truth_df_row = get_tracks(df)

    # range of search for slope (m) and intercept (b) values: bin size
    left_m_limit, right_m_limit = limits[0]
    left_b_limit, right_b_limit = limits[1]
    m_range = np.arange(left_m_limit, right_m_limit, bin_size[0])
    b_range = np.arange(left_b_limit, right_b_limit, bin_size[1])

    # accumulator array that will store the scores (votes) of the tracks
    accumulator = np.zeros((m_range.shape[0], b_range.shape[0]))

    # dictionary that stores which tracks go through each bin
    tracks_per_bin = {}

    # loop for all the tracks in the Hough space
    for idx, track_1 in enumerate(all_tracks):

        # find intersections with all the other tracks
        for track_2 in all_tracks[idx+1:]:
            
            # find the intersection point of the two tracks
            intersection_x, intersection_y = find_intersections(track_1, track_2)

            # if the intersection point is outside of the boundary, ignore it
            if (not left_m_limit <= intersection_x <= right_m_limit) or not (left_b_limit <= intersection_y <= right_b_limit):
                continue

            # get the bin it belongs to
            bin_x, bin_y = find_bin(intersection_x, intersection_y, left_m_limit, left_b_limit, bin_size)

            # increment the accumulator for that bin and add the tracks to the bin
            accumulator[bin_x, bin_y] += 1
            update_tracks_per_bin(tracks_per_bin, bin_x, bin_y, track_1, track_2)

    # remove duplicate tracks from inside the same bin
    remove_duplicate_tracks(tracks_per_bin)

    # get the indices of the bins with the most hits and split in r/z-coordinates
    top_indices = largest_indices(accumulator, num_tracks_to_estimate)
    bin_xs, bin_ys = top_indices

    # compute the coefficients (hence the tracks) from the location of the bin and book-keep which hits it contains
    est_tracks_to_hits = {}
    for bin_x, bin_y in zip(bin_xs, bin_ys):
        key_m = left_m_limit + bin_x * bin_size[0]
        key_b = left_b_limit + bin_y * bin_size[1]
        est_tracks_to_hits[(key_m, key_b)] = tracks_per_bin[(bin_x, bin_y)]

    return est_tracks_to_hits, hit_to_truth_df_row
