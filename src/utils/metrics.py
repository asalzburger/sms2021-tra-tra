import pandas as pd
from tqdm.notebook import tqdm
from src.utils.preprocessing import get_track_to_truth_row_mapping


def compute_sum_of_weights(hits, track_to_truth_df_row, truth_df):
    """ Returns a dictionary that maps a particle to the sum of the weights
        of its hits present in the `hits` list. """
    sum_of_weights_per_particle = {}
    for hit in hits:
        row = track_to_truth_df_row[hit]
        df_hit_row = truth_df.iloc[row, :]
        pid, weight = df_hit_row['particle_id'], df_hit_row['weight']
        sum_of_weights_per_particle[pid] = \
            sum_of_weights_per_particle.get(pid, 0.0) + weight
    return sum_of_weights_per_particle


def leading_particle(hits, track_to_truth_df_row, truth_df):
    """ Returns the particle ID of the particle with the
        highest sum of weights of individual hits. """
    weight_sums = compute_sum_of_weights(hits, track_to_truth_df_row, truth_df)
    return max(weight_sums, key=weight_sums.get)


def matching_probability(hits, track_to_truth_df_row, truth_df):
    """
    :param list[tuple[float, float]] hits:
        A list with all the hits that correspond (i.e. belong) to a specific
        estimated track.

    :param dict[tuple[float, float], int] track_to_truth_df_row:
        A dictionary that maps a track to the truth-dataframe row
        that corresponds to it.
        
    :param pd.DataFrame truth_df:
        A pandas dataframe that contains the truth values
        for every hit and every particle.


    :returns:
        leading-particle,
        sum{weights-of-hits-found-for-leading-particle} /
            sum{weights-of-all-hits-of-leading-particle}
    :rtype:
        tuple[str, double]
    """
    # get the leading particle
    leading_pid = leading_particle(hits, track_to_truth_df_row, truth_df)
    leading_particle_df = truth_df[truth_df['particle_id'] == leading_pid]
    
    # compute the sum of weights for the hits of the leading particle found
    # and the total sum of weights
    hits_found_weight = 0.0
    total_weight = 0.0
    for _, series in leading_particle_df.iterrows():
        track = series['track']
        weight = series['weight']
        if track in hits:
            hits_found_weight += weight
        total_weight += weight
        
    # return the percentage
    return leading_pid, hits_found_weight / total_weight


def purity(hits, track_to_truth_df_row, truth_df):
    """
    :param list[tuple[float, float]] hits:
        A list with all the hits that correspond (i.e. belong) to a specific
        estimated track.

    :param dict[tuple[float, float], int] track_to_truth_df_row:
        A dictionary that maps a track to the truth-dataframe row
        that corresponds to it.

    :param pd.DataFrame truth_df:
        A pandas dataframe that contains the truth values
        for every hit and every particle.


    :returns:
        leading-particle,
        sum{weights-of-hits-found-for-leading-particle} /
            sum{weights-of-all-hits-found}
    :rtype:
        tuple[str, float]
    """
    weight_sums = compute_sum_of_weights(hits, track_to_truth_df_row, truth_df)
    leading_pid = max(weight_sums, key=weight_sums.get)
    total_weight = sum(weight_sums.values())
    return leading_pid, weight_sums[leading_pid] / total_weight


def efficiency_rate(hits_per_estimated_track, truth_df, threshold=0.5):
    """
    :param list[list[tuple[float, float]]] hits_per_estimated_track:
        A list that contains lists with all the hits that correspond
        (i.e. belong) to a specific estimated track.

    :param pd.DataFrame truth_df:
        A pandas dataframe that contains the truth values for every hit
        and every particle.

    :param float threshold:
        A threshold that if achieved, then the estimated track is considered to
        have successfully reconstructed a particle track.

    
    :returns:
        # {reconstructed tracks} / # {total number of existing tracks}
    :rtype:
        float
    """
    num_true_particles = len(set(truth_df['particle_id']))
    track_to_truth_df_row = get_track_to_truth_row_mapping(truth_df)
    found_particles = set()
    for hits in tqdm(hits_per_estimated_track, desc='Efficiency Rate'):
        pid, prob = matching_probability(hits, track_to_truth_df_row, truth_df)
        if prob >= threshold:
            found_particles.add(pid)
        if len(found_particles) == num_true_particles:
            break
    
    return len(found_particles) / len(set(truth_df['particle_id']))


def fake_rate(hits_per_estimated_track, truth_df, threshold=0.25):
    """
    :param list[list[tuple[float, float]]] hits_per_estimated_track:
        A list that contains lists with all the hits that correspond
        (i.e. belong) to a specific estimated track.

    :param pd.DataFrame truth_df:
        A pandas dataframe that contains the truth values for every hit
        and every particle.

    :param float threshold:
        The minimum percentage of hits from the leading particle that must be
        present in all the hits of an estimated track.


    :returns:
        # {fake tracks found} / # {total number of estimated tracks}
    :rtype:
        double
    """
    track_to_truth_df_row = get_track_to_truth_row_mapping(truth_df)
    count_below_threshold = 0
    for hits in tqdm(hits_per_estimated_track, desc='Fake Rate'):
        _, prob = matching_probability(hits, track_to_truth_df_row, truth_df)
        if prob < threshold:
            count_below_threshold += 1

    return count_below_threshold / len(hits_per_estimated_track) \
        if len(hits_per_estimated_track) > 0 \
        else 0.0


def duplicate_rate(hits_per_estimated_track, truth_df):
    """
    :param list[list[tuple[float, float]]] hits_per_estimated_track:
        A list that contains lists with all the hits that correspond
        (i.e. belong) to a specific estimated track.

    :param pd.DataFrame truth_df:
        A pandas dataframe that contains the truth values for every hit
        and every particle.


    :returns:
        1 - # {different particles found} / # {total number of estimated tracks}
    :rtype:
        double
    """
    track_to_truth_df_row = get_track_to_truth_row_mapping(truth_df)
    particles_found = [leading_particle(hits, track_to_truth_df_row, truth_df)
                       for hits in tqdm(hits_per_estimated_track,
                                        desc='Duplicate Rate')]
    return (1.0 - len(set(particles_found)) / len(hits_per_estimated_track)) \
        if len(hits_per_estimated_track) > 0 \
        else 0.0
