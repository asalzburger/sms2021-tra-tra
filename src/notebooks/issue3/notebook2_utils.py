import numpy as np
import pandas as pd

from operator import itemgetter



def get_tracks(df):
    """ Returns a list of all the truth hits (which become tracks in the hough space). """
    return [(-x_i / y_i, (x_i ** 2 + y_i ** 2) / (2 * y_i)) for x_i, y_i in zip(df['tx'], df['ty'])]


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
            tracks_to_intersections[(idx_1, idx_2)] = (intersection_x, intersection_y)
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
    """
    x_range = np.arange(x_limits[0], x_limits[1], bin_size[0])
    y_range = np.arange(y_limits[0], y_limits[1], bin_size[1])

    # accumulator array that will store the scores (votes) of the tracks
    accumulator = np.zeros((x_range.shape[0], y_range.shape[0]))
    """

    # dictionary that stores which tracks go through each bin
    tracks_per_bin = {}

    # loop for all the tracks in the Hough space
    for idx_1, track_1 in enumerate(tracks):
        for idx_2, track_2 in enumerate(tracks[idx_1 + 1:]):
            
            # retrieve the intersection point of the two tracks
            intersection_x, intersection_y = intersections[(idx_1, idx_2)]
            # get the bin it belongs to
            bin_x, bin_y = find_bin(intersection_x, intersection_y, x_limits[0], y_limits[0], bin_size)

            # increment the accumulator for that bin and add the tracks to the bin
            # accumulator[bin_x, bin_y] += 1
            update_tracks_per_bin(tracks_per_bin, bin_x, bin_y, track_1, track_2)

    # remove duplicate tracks from inside the same bin
    remove_duplicate_tracks(tracks_per_bin)

    # get the indices of the bins that have at least the minimum required hits
    top_indices = bins_with_least_hits(tracks_per_bin, selection_hyperparameters['minimum-hits-per-bin'])

    # compute the coefficients (hence the tracks) from the location of the bin and book-keep which hits it contains
    est_tracks_to_hits = {}
    for bin_x, bin_y in top_indices:
        key_m = x_limits[0] + bin_x * bin_size[0]
        key_b = y_limits[0] + bin_y * bin_size[1]
        est_tracks_to_hits[(key_m, key_b)] = tracks_per_bin[(bin_x, bin_y)]

    return est_tracks_to_hits


def get_truth_hit_mapping(df):
    """ Returns a dictionary that maps a hit (track in the Hough space) to the df it belongs to. """
    hit_to_truth_df_row = {}
    for row, series in df.iterrows():
        x_i = series['tx']
        y_i = series['ty']
        hit = (-x_i / y_i, (x_i ** 2 + y_i ** 2) / (2 * y_i))
        hit_to_truth_df_row[hit] = row
    
    return hit_to_truth_df_row


def leading_particle(hits, hit_to_truth_df_row, truth_df):
    """ Returns the particle ID of the particle with the highest sum of weights of individual hits. """
    
    sum_of_weights_per_particle = {}
    for hit in hits:
        row = hit_to_truth_df_row[hit]
        df_hit_row = truth_df.iloc[row, :]
        pid, weight = df_hit_row['particle_id'], df_hit_row['weight']
        sum_of_weights_per_particle[pid] = sum_of_weights_per_particle.get(pid, 0.0) + weight
        
    return max(sum_of_weights_per_particle, key=sum_of_weights_per_particle.get)


def matching_probability(hits, truth_df):
    """
    :param list[tuple[float, float]] hits:
        A list with all the hits that correspond (i.e. belong) to a specific estimated track.
        
    :param dict[tuple[float, float], int] hit_to_truth_df_row:
        A dictionary that maps a hit to the row of the truth dataframe that contains that hit.
        
    :param pd.DataFrame truth_df:
        A pandas dataframe that contains the truth values for every hit and every particle.


    :returns:
        \sum{weights_of_hits_found_for_leading_particle} / \sum{weights_of_all_hits_of_leading_particle}
    :rtype:
        double
    """
    # (track in Hough space) -> row of hit in dataframe
    hit_to_truth_df_row = get_truth_hit_mapping(truth_df)

    # get the leading particle
    leading_particle_id = leading_particle(hits, hit_to_truth_df_row, truth_df)
    leading_particle_df = truth_df[truth_df['particle_id'] == leading_particle_id]
    
    # compute the sum of weights for the hits of the leading particle found, and the total sum of weights
    hits_found_weight = 0.0
    total_weight = 0.0
    for _, series in leading_particle_df.iterrows():
        x_i = series['tx']
        y_i = series['ty']
        hit = (-x_i / y_i, (x_i ** 2 + y_i ** 2) / (2 * y_i))
        weight = series['weight']
        if hit in hits:
            hits_found_weight += weight
        total_weight += weight
        
    # return the percentage
    return hits_found_weight / total_weight


def efficiency_rate(hits_per_estimated_track, truth_df, threshold=0.5):
    """
    :param list[list[tuple[float, float]]] hits_per_estimated_track:
        A list that contains lists with all the hits that correspond (i.e. belong) to a specific estimated track.

    :param pd.DataFrame truth_df:
        A pandas dataframe that contains the truth values for every hit and every particle.

    :param float threshold:
        A threshold that if achieved, then the estimated track is considered to have
        succesfully reconstructed a particle track.

    
    :returns:
        # {successfully reconstructed tracks} / # {estimated tracks}
    :rtype:
        float
    """
    count_above_threshold = 0
    for hits in hits_per_estimated_track:
        if matching_probability(hits, truth_df) >= threshold:
            count_above_threshold += 1
    
    return count_above_threshold / len(hits_per_estimated_track)


# ToDo: Cross check to see if this is ok
def fake_rate(hits_per_estimated_track, truth_df, threshold=0.25):
    """
    :param list[list[tuple[float, float]]] hits_per_estimated_track:
        A list that contains lists with all the hits that correspond (i.e. belong) to a specific estimated track.

    :param pd.DataFrame truth_df:
        A pandas dataframe that contains the truth values for every hit and every particle.

    :param float threshold:
        The minimum percentage of hits from the leading particle that must be present in all the hits
        of an estimated track.


    :returns:
        # {fake tracks found} / # {total number of estimated tracks}
    :rtype:
        double
    """
    count_below_threshold = 0
    for hits in hits_per_estimated_track:
        if matching_probability(hits, truth_df) < threshold:
            count_below_threshold += 1

    return count_below_threshold / len(hits_per_estimated_track)


def duplicate_rate(hits_per_estimated_track, truth_df):
    """
    :param list[list[tuple[float, float]]] hits_per_estimated_track:
        A list that contains lists with all the hits that correspond (i.e. belong) to a specific estimated track.

    :param pd.DataFrame truth_df:
        A pandas dataframe that contains the truth values for every hit and every particle.


    :returns:
        1 - # {different particles found} / # {total number of estimated tracks}
    :rtype:
        double
    """
    hit_to_truth_df_row = get_truth_hit_mapping(truth_df)
    particles_found = [leading_particle(hits, hit_to_truth_df_row, truth_df) for hits in hits_per_estimated_track]
    return 1.0 - len(set(particles_found)) / len(hits_per_estimated_track)