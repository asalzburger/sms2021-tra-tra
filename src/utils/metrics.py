import numpy as np
import pandas as pd


def leading_particle(hits, hit_to_particle, threshold=None):
    """ Returns the particle ID of the particle with the most hits in the "hits" list. """
    number_of_hits_per_particle = {}
    for hit in hits:
        particle_id = hit_to_particle[hit]
        number_of_hits_per_particle[particle_id] = number_of_hits_per_particle.get(particle_id, 0) + 1
        
    _leading_particle = max(number_of_hits_per_particle, key=number_of_hits_per_particle.get)

    # if a threshold is specified, make sure the leading satisfies it
    if threshold is not None and number_of_hits_per_particle[_leading_particle] / len(hits) < threshold:
        _leading_particle = None
    return _leading_particle


def efficiency_rate(hits, hit_to_particle, truth_df):
    """
    :param list[tuple[float, float]] hits:
        A list with all the hits that correspond (i.e. belong) to a specific estimated track.
        
    :param dict[tuple[float, float], str] hit_to_particle:
        A dictionary that maps a hit to the particle ID that produced it.
        
    :param pd.DataFrame truth_df:
        A pandas dataframe that contains the truth values for every hit and every particle.


    :returns:
        \sum{weights_of_hits_found_for_leading_particle} / \sum{weights_of_all_hits_of_leading_particle}
    :rtype:
        double
    """

    # get the leading particle
    leading_particle_id = leading_particle(hits, hit_to_particle)
    leading_particle_df = truth_df[truth_df['particle_id'] == leading_particle_id]
    
    # compute the sum of weights for the hits of the leading particle found, and the total sum of weights
    hits_found_weight = 0.0
    total_weight = 0.0
    for row in leading_particle_df.iterrows():
        hit = -row[1]['r'], row[1]['tz']
        weight = row[1]['weight']
        if hit in hits:
            hits_found_weight += weight
        total_weight += weight
        
    # return the percentage
    return hits_found_weight / total_weight


# ToDo: Cross check to see if this is ok
def fake_rate(hits_per_estimated_track, hit_to_particle, threshold=0.5):
    """
    :param list[list[tuple[float, float]]] hits_per_estimated_track:
        A list that contains lists with all the hits that correspond (i.e. belong) to a specific estimated track.

    :param dict[tuple[float, float], str] hit_to_particle:
        A dictionary that maps a hit to the particle ID that produced it.

    :param float threshold:
        The minimum percentage of hits from the leading particle that must be present in all the hits
        of an estimated track.


    :returns:
        # {fake tracks found} / # {total number of estimated tracks}
    :rtype:
        double
    """
    leading_particle_per_track = [leading_particle(hits, hit_to_particle, threshold=threshold)
                                  for hits in hits_per_estimated_track]
    return leading_particle_per_track.count(None) / len(hits_per_estimated_track)


def duplicate_rate(hits_per_estimated_track, hit_to_particle):
    """
    :param list[list[tuple[float, float]]] hits_per_estimated_track:
        A list that contains lists with all the hits that correspond (i.e. belong) to a specific estimated track.

    :param dict[tuple[float, float], str] hit_to_particle:
        A dictionary that maps a hit to the particle ID that produced it.


    :returns:
        # {different particles found} / # {total number of estimated tracks}
    :rtype:
        double
    """
    particles_found = [leading_particle(hits, hit_to_particle) for hits in hits_per_estimated_track]
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
