import numpy as np
import pandas as pd



def leading_particle(hits, hit_to_truth_df_row, truth_df):
    """ Returns the particle ID of the particle with the highest sum of weights of individual hits. """
    
    sum_of_weights_per_particle = {}
    for hit in hits:
        row = hit_to_truth_df_row[hit]
        df_hit_row = truth_df.iloc[row, :]
        pid, weight = df_hit_row['particle_id'], df_hit_row['weight']
        sum_of_weights_per_particle[pid] = sum_of_weights_per_particle.get(pid, 0.0) + weight
        
    return max(sum_of_weights_per_particle, key=sum_of_weights_per_particle.get)


def matching_probability(hits, hit_to_truth_df_row, truth_df):
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

    # get the leading particle
    leading_particle_id = leading_particle(hits, hit_to_truth_df_row, truth_df)
    leading_particle_df = truth_df[truth_df['particle_id'] == leading_particle_id]
    
    # compute the sum of weights for the hits of the leading particle found, and the total sum of weights
    hits_found_weight = 0.0
    total_weight = 0.0
    for _, series in leading_particle_df.iterrows():
        hit = -series['r'], series['tz']
        weight = series['weight']
        if hit in hits:
            hits_found_weight += weight
        total_weight += weight
        
    # return the percentage
    return hits_found_weight / total_weight


def efficiency_rate(hits_per_estimated_track, hit_to_truth_df_row, truth_df, threshold=0.5):
    """
    :param list[list[tuple[float, float]]] hits_per_estimated_track:
        A list that contains lists with all the hits that correspond (i.e. belong) to a specific estimated track.

    :param dict[tuple[float, float], int] hit_to_truth_df_row:
        A dictionary that maps a hit to the row of the truth dataframe that contains that hit.
        
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
        if matching_probability(hits, hit_to_truth_df_row, truth_df) >= threshold:
            count_above_threshold += 1
    
    return count_above_threshold / len(hits_per_estimated_track)


# ToDo: Cross check to see if this is ok
def fake_rate(hits_per_estimated_track, hit_to_truth_df_row, truth_df, threshold=0.25):
    """
    :param list[list[tuple[float, float]]] hits_per_estimated_track:
        A list that contains lists with all the hits that correspond (i.e. belong) to a specific estimated track.

    :param dict[tuple[float, float], int] hit_to_truth_df_row:
        A dictionary that maps a hit to the row of the truth dataframe that contains that hit.

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
        if matching_probability(hits, hit_to_truth_df_row, truth_df) < threshold:
            count_below_threshold += 1

    return count_below_threshold / len(hits_per_estimated_track)


def duplicate_rate(hits_per_estimated_track, hit_to_truth_df_row, truth_df):
    """
    :param list[list[tuple[float, float]]] hits_per_estimated_track:
        A list that contains lists with all the hits that correspond (i.e. belong) to a specific estimated track.

    :param dict[tuple[float, float], int] hit_to_truth_df_row:
        A dictionary that maps a hit to the row of the truth dataframe that contains that hit.

    :param pd.DataFrame truth_df:
        A pandas dataframe that contains the truth values for every hit and every particle.


    :returns:
        1 - # {different particles found} / # {total number of estimated tracks}
    :rtype:
        double
    """
    particles_found = [leading_particle(hits, hit_to_truth_df_row, truth_df) for hits in hits_per_estimated_track]
    return 1.0 - len(set(particles_found)) / len(hits_per_estimated_track)



def is_effectively_same_track(track1, track2, m_threshold=0.001, b_threshold=0.05):
    """ Given some threshold, decides if two tracks are the same. """
    m1, b1, = track1
    m2, b2, = track2
    return np.abs(m1 - m2) < m_threshold and np.abs(b1 - b2) < b_threshold


def efficiency_rate_2(true_tracks, est_tracks):
    """
    :param list[tuple[float, float]] true_tracks:
        A list that contains the ground truth tracks of all the particles of an event simulation.

    :param list[tuple[float, float]] est_tracks:
        A list that contains estimated tracks.

    
    :returns:
        number_of_tracks_correctly_found / total_number_of_ground_truth_tracks
    :rtype:
        double
    """
    counter_track_found = 0
    for idx1, t1 in enumerate(true_tracks):
        for idx2, t2 in enumerate(est_tracks):
            if is_effectively_same_track(t1, t2):
                counter_track_found += 1
                del est_tracks[idx2]
                break
    
    return counter_track_found / len(true_tracks)



def fake_rate_2(true_tracks, est_tracks):
    """
    :param list[tuple[float, float]] true_tracks:
        A list that contains the ground truth tracks of all the particles of an event simulation.

    :param list[tuple[float, float]] est_tracks:
        A list that contains estimated tracks.

    
    :returns:
        number_of_fake_tracks_estimated / total_number_of_estimated_tracks
    :rtype:
        double
    """
    counter_fake_tracks = 0
    for t1 in est_tracks:
        flag_close_track_was_found = False
        for t2 in true_tracks:
            if is_effectively_same_track(t1, t2, m_threshold=0.005, b_threshold=0.05):
                flag_close_track_was_found = True
                break
        if flag_close_track_was_found is False:
            counter_fake_tracks += 1
            
    return counter_fake_tracks / len(est_tracks)



def duplicate_rate_2(est_tracks):
    """
    :param list[tuple[float, float]] est_tracks:
        A list that contains estimated tracks.

    :returns:
        number_of_duplicate_tracks_estimated / total_number_of_estimated_tracks
    :rtype:
        double
    """
    counter_duplicate_tracks = 0
    for idx, t1 in enumerate(est_tracks):
        for t2 in est_tracks[idx + 1:]:
            if is_effectively_same_track(t1, t2):
                counter_duplicate_tracks += 1
                break
    
    return counter_duplicate_tracks / len(est_tracks)
