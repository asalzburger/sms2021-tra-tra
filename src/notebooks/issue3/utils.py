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


def pipeline(df, bin_size, limits, num_tracks_to_estimate):
    """
    :param pd.DataFrame df:
        The dataframe with the hits information.

    :param tuple[float, float] bin_size:
        The size of each bin (from the binning process): (width, height).

    :param tuple[float, float] limits:
        The range of selection for the values of the parameters (m, b).

    :param int num_tracks_to_estimate:
        Number of tracks to sample from the accumulator.
        Later will remove this, as selection will occur with other methods.
    """
    # book-keeping: store which particles form which track in the Hough space
    all_tracks = []
    hit_to_truth_df_row = {}
    for row, series in df.iterrows():
        r_i, z_i = series['r'], series['tz']
        hit = (-r_i, z_i)
        hit_to_truth_df_row[hit] = row
        # a hit becomes a track in the Hough space
        all_tracks.append(hit)

    # range of search for slope (m) and intercept (b) values: bin size
    m_limit, b_limit = limits
    m_range = np.arange(-m_limit, m_limit, bin_size[0])
    b_range = np.arange(-b_limit, b_limit, bin_size[1])

    # accumulator array that will store the scores (votes) of the tracks
    accumulator = np.zeros((m_range.shape[0], b_range.shape[0]))

    # dictionary that stores which tracks go through each bin
    tracks_per_bin = {}

    # loop for all the tracks in the Hough space
    for idx, track_1 in enumerate(all_tracks):
        m1, b1 = track_1

        # find intersections with all the other tracks
        for track_2 in all_tracks[idx+1:]:
            m2, b2 = track_2
            
            # find the intersection point of the two tracks
            intersection_x = - (b2 - b1) / (m2 - m1)
            intersection_y = m1 * intersection_x + b1

            # if the intersection point is outside of the boundary, ignore it
            if (not -m_limit <= intersection_x <= m_limit) or not (-b_limit <= intersection_y <= b_limit):
                continue

            # reduce to 3 decimal digits
            reduced_x = truncate(intersection_x, 3)
            reduced_y = truncate(intersection_y, 3)

            # compute the exact bin that will contain the above (x, y) point
            bin_x = int((reduced_x + m_limit) * 1000)
            bin_y = int((reduced_y + b_limit) * 1000)

            # increment the accumulator for that bin and add the tracks to the bin
            accumulator[bin_x, bin_y] += 1
            if (bin_x, bin_y) not in tracks_per_bin:
                tracks_per_bin[(bin_x, bin_y)] = [track_1, track_2]
            else:
                tracks_per_bin[(bin_x, bin_y)].extend((track_1, track_2))


    # remove duplicate tracks from inside the same bin
    for _bin, tracks_in_it in tracks_per_bin.items():
        tracks_per_bin[_bin] = list(set(tracks_in_it))

    # get the indices of the bins with the most hits
    l = largest_indices(accumulator, num_tracks_to_estimate) ## param
    # split the list in x-coordinates (actually r-coordinates) and y-coordinates (actually z-coordinates)
    bin_xs, bin_ys = l

    # compute the coefficients (hence the tracks) from the location of the bin and book-keep which hits it contains
    est_tracks_to_hits = {
        (x / 1000 - m_limit, y / 1000 - b_limit): tracks_per_bin[(x, y)] for x, y in zip(bin_xs, bin_ys)
    }
    # est_tracks = list(est_tracks_to_hits)
    return est_tracks_to_hits, hit_to_truth_df_row
