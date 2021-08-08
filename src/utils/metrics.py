import numpy as np
import pandas as pd


def get_track_to_truth_row_mapping(truth_df):
    """ Returns a dictionary that maps a track to its row index in the dataframe. """
    return {series['track']: row for row, series in truth_df.iterrows()}


def leading_particle(hits, track_to_truth_df_row, truth_df):
    """ Returns the particle ID of the particle with the highest sum of weights of individual hits. """
    sum_of_weights_per_particle = {}
    for hit in hits:
        row = track_to_truth_df_row[hit]
        df_hit_row = truth_df.iloc[row, :]
        pid, weight = df_hit_row['particle_id'], df_hit_row['weight']
        sum_of_weights_per_particle[pid] = sum_of_weights_per_particle.get(pid, 0.0) + weight
        
    return max(sum_of_weights_per_particle, key=sum_of_weights_per_particle.get)


def matching_probability(hits, track_to_truth_df_row, truth_df):
    """
    :param list[tuple[float, float]] hits:
        A list with all the hits that correspond (i.e. belong) to a specific estimated track.

    :param dict[tuple[float, float], int] track_to_truth_df_row:
        A dictionary that maps a track to the tuth_dataframe row that corresponds to it.
        
    :param pd.DataFrame truth_df:
        A pandas dataframe that contains the truth values for every hit and every particle.


    :returns:
        leading-particle, \sum{weights-of-hits-found-for-leading-particle} / \sum{weights-of-all-hits-of-leading-particle}
    :rtype:
        tuple[str, double]
    """

    # get the leading particle
    leading_particle_id = leading_particle(hits, track_to_truth_df_row, truth_df)
    leading_particle_df = truth_df[truth_df['particle_id'] == leading_particle_id]
    
    # compute the sum of weights for the hits of the leading particle found, and the total sum of weights
    hits_found_weight = 0.0
    total_weight = 0.0
    for _, series in leading_particle_df.iterrows():
        track = series['track']
        weight = series['weight']
        if track in hits:
            hits_found_weight += weight
        total_weight += weight
        
    # return the percentage
    return leading_particle_id, hits_found_weight / total_weight


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
        # {successfully reconstructed tracks} / # {total number of existing tracks}
    :rtype:
        float
    """
    num_true_particles = len(set(truth_df['particle_id']))
    track_to_truth_df_row = get_track_to_truth_row_mapping(truth_df)
    found_particles = set()
    for hits in hits_per_estimated_track:
        pid, prob = matching_probability(hits, track_to_truth_df_row, truth_df)
        if prob >= threshold:
            found_particles.add(pid)
        if len(found_particles) == num_true_particles:
            break
    
    return len(found_particles) / len(set(truth_df['particle_id']))


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
    track_to_truth_df_row = get_track_to_truth_row_mapping(truth_df)
    count_below_threshold = 0
    for hits in hits_per_estimated_track:
        _, prob = matching_probability(hits, track_to_truth_df_row, truth_df)
        if prob < threshold:
            count_below_threshold += 1

    return count_below_threshold / len(hits_per_estimated_track) if len(hits_per_estimated_track) > 0 else 0.0


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
    track_to_truth_df_row = get_track_to_truth_row_mapping(truth_df)
    particles_found = [leading_particle(hits, track_to_truth_df_row, truth_df) for hits in hits_per_estimated_track]
    return (1.0 - len(set(particles_found)) / len(hits_per_estimated_track)) if len(hits_per_estimated_track) > 0 else 0.0



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
