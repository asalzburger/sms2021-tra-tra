from tqdm.notebook import tqdm


def leading_particle(hits, num_true_particles, pids, weights):
    """
    :param list[int] hits:
        A list with all the indices of hits that correspond (i.e. belong)
        to a specific estimated track.

    :param int num_true_particles:
        The number of different particles in an event (truth dataframe).

    :param list[int] pids:
        A list that contains all the indices of all (discrete) particles IDs.

    :param list[float] weights:
        A list that contains all the particles weights, with the order seen
        in the truth dataframe.


    :returns:
        leading-particle-index, sum{weights-of-hits-found-for-leading-particle},
        list-with-sum-of-weights-for-each-particle
    :rtype:
        tuple[int, float, list[float]]
    """
    # initialize counters and an accumulator
    leading_pid_idx, leading_pid_sum = -1, 0.0
    sum_of_weights_per_particle = [0.0] * num_true_particles

    # loop through each hit found
    for hit_idx in hits:

        # get the particle it belongs to and its weight
        discrete_pid_idx, weight = pids[hit_idx], weights[hit_idx]
        sum_of_weights_per_particle[discrete_pid_idx] += weight

        # update the counters
        if sum_of_weights_per_particle[discrete_pid_idx] > leading_pid_sum:
            leading_pid_sum = sum_of_weights_per_particle[discrete_pid_idx]
            leading_pid_idx = discrete_pid_idx

    return leading_pid_idx, leading_pid_sum, sum_of_weights_per_particle


def matching_probability(hits, pids, weights, pid_to_total_weight):
    """
    :param list[int] hits:
        A list with all the indices of hits that correspond (i.e. belong)
        to a specific estimated track.

    :param list[int] pids:
        A list that contains all the indices of all (discrete) particles IDs.

    :param list[float] weights:
        A list that contains all the hits weights, with the order seen
        in the truth dataframe.

    :param list[float] pid_to_total_weight:
        A list that maps a particle ID index to the sum of the weights of all
        its hits (in the truth dataframe).


    :returns:
        leading-particle,
        sum{weights-of-hits-found-for-leading-particle} /
            sum{weights-of-all-hits-of-leading-particle}
    :rtype:
        tuple[str, float]
    """
    # get the leading particle, all the weight sums and the total sum for it
    leading_pid_idx, leading_pid_present_weight, weight_sums = \
        leading_particle(hits, len(pid_to_total_weight), pids, weights)
    leading_pid_total_weight = pid_to_total_weight[leading_pid_idx]

    # return the leading particle ID and the percentage of weight present
    matching_prob = leading_pid_present_weight / leading_pid_total_weight
    return leading_pid_idx, matching_prob


def purity(hits, num_true_particles, pids, weights):
    """
    :param list[int] hits:
        A list with all the indices of hits that correspond (i.e. belong)
        to a specific estimated track.

    :param int num_true_particles:
        The number of different particles in an event (truth dataframe).

    :param list[int] pids:
        A list that contains all the indices of all (discrete) particles IDs.

    :param list[float] weights:
        A list that contains all the particles weights, with the order seen
        in the truth dataframe.


    :returns:
        leading-particle,
        sum{weights-of-hits-found-for-leading-particle} /
            sum{weights-of-all-hits-found}
    :rtype:
        tuple[str, float]
    """
    # get the leading particle, all the weight sums and the total sum of the bin
    leading_pid, leading_pid_present_weight, weight_sums = \
        leading_particle(hits, num_true_particles, pids, weights)
    total_weight_inside_bin = sum(weight_sums)

    # return the leading particle ID and the percentage of leading-pid-weight
    return leading_pid, leading_pid_present_weight / total_weight_inside_bin


def efficiency_rate(hits_per_est_track, pids, weights, pid_to_total_weight,
                    threshold=0.5):
    """
    :param list[list[int]] hits_per_est_track:
        A list that contains lists with all the hits that correspond
        (i.e. belong) to a specific estimated track.

    :param list[int] pids:
        A list that contains all the indices of all (discrete) particles IDs.

    :param list[float] weights:
        A list that contains all the particles weights, with the order seen
        in the truth dataframe.

    :param list[float] pid_to_total_weight:
        A list that maps a particle ID index to the sum of the weights of all
        its hits (in the truth dataframe).

    :param float threshold:
        A threshold that if achieved, then the estimated track is considered to
        have successfully reconstructed a particle track.


    :returns:
        # {reconstructed tracks} / # {total number of existing tracks}
    :rtype:
        float
    """
    num_true_particles = len(pid_to_total_weight)
    num_particles_found = 0
    found_particles = [0] * num_true_particles

    for hits in tqdm(hits_per_est_track, desc='Efficiency Rate'):
        pid_idx, prob = matching_probability(hits, pids, weights,
                                             pid_to_total_weight)
        if found_particles[pid_idx] == 0 and prob >= threshold:
            found_particles[pid_idx] = 1
            num_particles_found += 1
        if num_particles_found == num_true_particles:
            break

    return num_particles_found / num_true_particles


def fake_rate(hits_per_est_track, pids, weights, pid_to_total_weight,
              threshold=0.25):
    """
    :param list[list[int]] hits_per_est_track:
        A list that contains lists with all the hits that correspond
        (i.e. belong) to a specific estimated track.

    :param list[int] pids:
        A list that contains all the indices of all (discrete) particles IDs.

    :param list[float] weights:
        A list that contains all the particles weights, with the order seen
        in the truth dataframe.

    :param list[float] pid_to_total_weight:
        A list that maps a particle ID index to the sum of the weights of all
        its hits (in the truth dataframe).

    :param float threshold:
        The minimum percentage of hits from the leading particle that must be
        present in all the hits of an estimated track.


    :returns:
        # {fake tracks found} / # {total number of estimated tracks}
    :rtype:
        float
    """
    num_estimated_particles = len(hits_per_est_track)
    if num_estimated_particles == 0:
        return 0.0

    count_below_threshold = 0
    for hits in tqdm(hits_per_est_track, desc='Fake Rate'):
        _, prob = matching_probability(hits, pids, weights,
                                       pid_to_total_weight)
        if prob < threshold:
            count_below_threshold += 1

    return count_below_threshold / num_estimated_particles


def duplicate_rate(hits_per_est_track, pids, weights, pid_to_total_weight):
    """
    :param list[list[int]] hits_per_est_track:
        A list that contains lists with all the hits that correspond
        (i.e. belong) to a specific estimated track.

    :param list[int] pids:
        A list that contains all the indices of all (discrete) particles IDs.

    :param list[float] weights:
        A list that contains all the particles weights, with the order seen
        in the truth dataframe.

    :param list[float] pid_to_total_weight:
        A list that maps a particle ID index to the sum of the weights of all
        its hits (in the truth dataframe).


    :returns:
        1 - # {particles found} / # {total number of estimated tracks}
    :rtype:
        float
    """
    num_true_particles = len(pid_to_total_weight)
    num_estimated_particles = len(hits_per_est_track)
    if num_estimated_particles == 0:
        return 0.0

    found_particles = [0] * num_true_particles
    num_particles_found = 0
    for hits in tqdm(hits_per_est_track, desc='Duplicate Rate'):
        pid = leading_particle(hits, num_true_particles, pids, weights)[0]
        if found_particles[pid] == 0:
            found_particles[pid] = 1
            num_particles_found += 1
        if num_particles_found == num_true_particles:
            break

    return 1.0 - num_particles_found / num_estimated_particles
