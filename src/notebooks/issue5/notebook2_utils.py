import math
import os

import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.abspath(os.path.join('..', '..', 'utils')))

from metrics import get_track_to_truth_row_mapping, leading_particle, matching_probability, efficiency_rate, fake_rate, duplicate_rate


def get_str(x, n=2):
    """ returns a float as a string and cuts at the n-th decimal digit """
    x = truncate(x, n)
    x = 0.0 if x == 0.0 else x
    before, after = str(x).split('.', 1)
    return '.'.join([before, after[:n]])


def truncate(f, n):
    """ Truncates the digits of a floating point number. e.g.: f(0.8756343, 3) = 0.875 """
    return math.floor(f * 10 ** n) / 10 ** n


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
            tracks_to_intersections[(idx_1, idx_1 + 1 + idx_2)] = (intersection_x, intersection_y)
    return tracks_to_intersections


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


def xy_pipeline(tracks, xy_hyperparams):
    """ Runs the phi-qpT Hough Transform algorithm to get the estimated tracks. """
    bin_size = xy_hyperparams['bin-size']
    width_limits, height_limits = xy_hyperparams['phi-range'], xy_hyperparams['qpt-range']
    nhits = xy_hyperparams['minimum-hits-per-bin']
    A = 3e-4

    # range of search
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
        ys = np.sin(xs - phi) / (A * r) if xy_hyperparams['use-sin'] else (xs - phi) / (A * r)

        # find in which bins the points belong to
        bin_xs, bin_ys = vect_find_bin(xs, ys, width_limits[0], height_limits[0], bin_size[0], bin_size[1])
        # flatten the indices in order to update the flat accumulator (which will automatically update the original accumulator)
        flat_indices = np.ravel_multi_index((bin_xs, bin_ys), accumulator.shape)
        flat_accumulator[flat_indices] += 1

        # update the book-keeping structure
        update_tracks_per_bin(tracks_per_bin, track, bin_xs, bin_ys)

    # get the indices of the bins that have at least the minimum required hits
    top_indices = bins_with_least_hits(tracks_per_bin, nhits)

    # compute the coefficients (hence the tracks) from the location of the bin and book-keep which hits it contains
    est_tracks_to_hits = {}
    for bin_x, bin_y in top_indices:
        key_x = width_limits[0] + bin_x * bin_size[0]
        key_y = height_limits[0] + bin_y * bin_size[1]
        est_tracks_to_hits[(key_x, key_y)] = tracks_per_bin[(bin_x, bin_y)]

    return accumulator, est_tracks_to_hits


def rz_pipeline(tracks, intersections, rz_hyperparams):
    """ Runs the r-z Hough Transform algorithm to get the estimated tracks. """
    # range of search for r and z values
    bin_size = rz_hyperparams['bin-size']
    r_limits, z_limits = rz_hyperparams['r-range'], rz_hyperparams['z-range']
    nhits = rz_hyperparams['minimum-hits-per-bin']

    # dictionary that stores which tracks go through each bin
    tracks_per_bin = {}

    # loop for all the tracks in the Hough space
    for idx_1, track_1 in enumerate(tracks):
        for idx_2, track_2 in enumerate(tracks[idx_1 + 1:]):
            
            # retrieve the intersection point of the two tracks
            intersection_r, intersection_z = intersections[(idx_1, idx_1 + 1 + idx_2)]

            # get the bin it belongs to
            if r_limits[0] <= intersection_r <= r_limits[1] and z_limits[0] <= intersection_z <= z_limits[1]:
                bin_x, bin_y = find_bin(intersection_r, intersection_z, r_limits[0], z_limits[0], bin_size[0], bin_size[1])

                # add the tracks to the bin
                if (bin_x, bin_y) not in tracks_per_bin:
                    tracks_per_bin[(bin_x, bin_y)] = [track_1, track_2]
                else:
                    tracks_per_bin[(bin_x, bin_y)].extend((track_1, track_2))

    # remove duplicate tracks from inside the same bin
    remove_duplicate_tracks(tracks_per_bin)

    # get the indices of the bins that have at least the minimum required hits
    top_indices = bins_with_least_hits(tracks_per_bin, nhits)

    # compute the coefficients (hence the tracks) from the location of the bin and book-keep which hits it contains
    est_tracks_to_hits = {}
    for bin_x, bin_y in top_indices:
        key_x = r_limits[0] + bin_x * bin_size[0]
        key_y = z_limits[0] + bin_y * bin_size[1]
        est_tracks_to_hits[(key_x, key_y)] = tracks_per_bin[(bin_x, bin_y)]

    return est_tracks_to_hits


def custom_efficiency_rate(est_tracks_to_hits, truth_df, threshold=0.5):
    """ Custom efficiency rate function. The difference is that it also returns the reconstructed particles. """
    num_true_particles = len(set(truth_df['particle_id']))
    track_to_truth_df_row = get_track_to_truth_row_mapping(truth_df)
    found_particles = set()
    pid_to_est_track = {}
    
    for track, hits in est_tracks_to_hits.items():
        pid, prob = matching_probability(hits, track_to_truth_df_row, truth_df)
        if prob >= threshold and pid not in found_particles:
            found_particles.add(pid)
            pid_to_est_track[pid] = track
        if len(found_particles) == num_true_particles:
            break

    return pid_to_est_track, len(found_particles) / len(set(truth_df['particle_id']))


def custom_fake_rate(est_tracks_to_hits, truth_df, threshold=0.25):
    """ Custom fake rate function. The difference is that it also returns the fake estimated tracks. """
    fakes = []
    track_to_truth_df_row = get_track_to_truth_row_mapping(truth_df)
    for track, hits in est_tracks_to_hits.items():
        _, prob = matching_probability(hits, track_to_truth_df_row, truth_df)
        if prob < threshold:
            fakes.append(track)

    return fakes, len(fakes) / len(est_tracks_to_hits)


def remove_found_hits(truth_df, est_tracks_to_hits):
    """ Removes the hits that were identified by the hough transform algorithm """
    df_copy = truth_df.copy()
    for track, hits in est_tracks_to_hits.items():
        for hit in hits:
            df_copy = df_copy[df_copy['track'] != hit]
    return df_copy


def combine_transforms_1(truth_df, xy_hyperparams, rz_hyperparams):
    """ Combines both the Helical (phi-q/p_T) and longitudinal (r-z) hough transforms to increase accuracy. """
    # counters to assess final rates
    num_total_particles = len(set(truth_df['particle_id']))
    num_particles_reco = 0
    num_fake_tracks = 0
    num_tracks_reco = 0
    
    # get the tracks run the pipeline to get the estimations of the phi-qpt transform
    all_xy_tracks = list(truth_df['xy_track'])
    _, est_tracks_to_hits = xy_pipeline(all_xy_tracks, xy_hyperparams)
    
    # get truth-particles found and update the counters that will be used to compute the aggregate rates
    truth_df['track'] = truth_df['xy_track']
    pid_to_est_track, _ = custom_efficiency_rate(est_tracks_to_hits, truth_df)
    num_particles_reco += len(pid_to_est_track)
    fakes, _ = custom_fake_rate(est_tracks_to_hits, truth_df)
    num_fake_tracks += len(fakes)
    num_tracks_reco += len(est_tracks_to_hits)
    
    # remove the hits that were found
    new_df = remove_found_hits(truth_df, est_tracks_to_hits)
    
    # get the remaining tracks and run the pipeline to get the estimations of the r-z transform
    all_rz_tracks = list(new_df['rz_track'])
    rz_intersections = compute_all_intersections(all_rz_tracks)
    est_tracks_to_hits = rz_pipeline(all_rz_tracks, rz_intersections, rz_hyperparams)
    
    # get the number of new particles found and aggregate to get the final rates
    truth_df['track'] = truth_df['rz_track']
    pid_to_est_track, _ = custom_efficiency_rate(est_tracks_to_hits, truth_df)
    num_particles_reco += len(pid_to_est_track)
    fakes, _ = custom_fake_rate(est_tracks_to_hits, truth_df)
    num_fake_tracks += len(fakes)
    num_tracks_reco += len(est_tracks_to_hits)

    # computes the rates and return them
    aggregated_efficiency_rate = num_particles_reco / num_total_particles
    aggregated_fake_rate = num_fake_tracks / num_tracks_reco
    aggregated_duplicate_rate = 1 - num_particles_reco / num_tracks_reco
    return aggregated_efficiency_rate, aggregated_fake_rate, aggregated_duplicate_rate


def common_hits_percentage(truth_df, hits1, hits2, xy_first=True):
    """ Returns the percentage of hits present in the first list of hits
        that are also present in the second, along with the common hits dataframe. """

    if xy_first:
        df_with_hits1 = truth_df[truth_df['xy_track'].isin(hits1)]
        df_with_hits2 = truth_df[truth_df['rz_track'].isin(hits2)]
    else:
        df_with_hits1 = truth_df[truth_df['rz_track'].isin(hits1)]
        df_with_hits2 = truth_df[truth_df['xy_track'].isin(hits2)]

    index1 = df_with_hits1.index.tolist()
    index2 = df_with_hits2.index.tolist()
    
    common_hits_idx = set(index1) & set(index2)
    common_percentage = len(common_hits_idx) / len(index1) if len(index1) > 0 else 0
    common_df = truth_df.iloc[list(common_hits_idx)]
    
    return common_percentage, common_df


def intersect_estimations(truth_df, est1, est2, threshold=0.5):
    """ Filters out tracks from both transforms by checking if the other transform
        has a bin with a high percentage of same particles in it. """
    
    def filter_out_estimations(truth_df, _est1, _est2, xy_first=True,_threshold=0.5):
        """ Assign to the `final_est` dictionary the key-values pairs of the `_est1`
            dictionary that also exist in a bin (a high percentage) in `_est2`. """
        final_est = {}
        for track1, hits1 in _est1.items():
            for hits2 in _est2.values():

                percentage_common, _ = common_hits_percentage(truth_df, hits1, hits2, xy_first=xy_first)
                if percentage_common >= _threshold:
                    final_est[track1] = hits1
                    break
        
        return final_est
    
    final_est1 = filter_out_estimations(truth_df, est1, est2, xy_first=True,_threshold=threshold)
    final_est2 = filter_out_estimations(truth_df, est2, est1, xy_first=False, _threshold=threshold)

    return final_est1, final_est2


def combine_transforms_2(truth_df, xy_hyperparams, rz_hyperparams):
    """ Combines both the Helical (phi-q/p_T) and longitudinal (r-z) hough transforms to decrease fake-duplicate rates. """
    # get the tracks run the pipeline to get the estimations of the phi-qpt transform
    all_xy_tracks = list(truth_df['xy_track'])
    _, xy_est_tracks_to_hits = xy_pipeline(all_xy_tracks, xy_hyperparams)
    
    # do the same for the r-z transform
    all_rz_tracks = list(truth_df['rz_track'])
    rz_intersections = compute_all_intersections(all_rz_tracks)
    rz_est_tracks_to_hits = rz_pipeline(all_rz_tracks, rz_intersections, rz_hyperparams)

    # get the intersection dictionaries
    xy_est_tracks_to_hits, rz_est_tracks_to_hits = intersect_estimations(truth_df, xy_est_tracks_to_hits, rz_est_tracks_to_hits)

    # compute the rates for every "improved" transform and then return them
    truth_df['track'] = truth_df['xy_track']
    xy_eff = efficiency_rate(xy_est_tracks_to_hits.values(), truth_df)
    xy_fake = fake_rate(xy_est_tracks_to_hits.values(), truth_df)
    xy_dup = duplicate_rate(xy_est_tracks_to_hits.values(), truth_df)

    truth_df['track'] = truth_df['rz_track']
    rz_eff = efficiency_rate(rz_est_tracks_to_hits.values(), truth_df)
    rz_fake = fake_rate(rz_est_tracks_to_hits.values(), truth_df)
    rz_dup = duplicate_rate(rz_est_tracks_to_hits.values(), truth_df)
    
    return (xy_eff, xy_fake, xy_dup), (rz_eff, rz_fake, rz_dup)


def unite_estimations(truth_df, est1, est2, threshold=0.5):
    """ Extends the hits per estimated track for a given transform, by looking at the other transform. """

    def extend_estimations(truth_df, _est1, _est2, xy_first=True,_threshold=0.5):
        """ Assigns unions of hits (for bins with common hits) in the `final_est` dictionary. """
        final_est = {}
        for track1, hits1 in _est1.items():
            for hits2 in _est2.values():

                percentage_common, _ = common_hits_percentage(truth_df, hits1, hits2, xy_first=xy_first)
                if percentage_common >= _threshold:
                    union_hits = set(hits1)

                    if xy_first:
                        df_with_hits2 = truth_df[truth_df['rz_track'].isin(hits2)]
                        hits2_in_other_transform = set(df_with_hits2['xy_track'])
                    else:
                        df_with_hits2 = truth_df[truth_df['xy_track'].isin(hits2)]
                        hits2_in_other_transform = set(df_with_hits2['rz_track'])
                    
                    union_hits.update(hits2_in_other_transform)
                    final_est[track1] = list(union_hits)
                    break

        return final_est
    
    final_est1 = extend_estimations(truth_df, est1, est2, xy_first=True,_threshold=threshold)
    final_est2 = extend_estimations(truth_df, est2, est1, xy_first=False, _threshold=threshold)

    return final_est1, final_est2


def combine_transforms_3(truth_df, xy_hyperparams, rz_hyperparams):
    """ Combines both the Helical (phi-q/p_T) and longitudinal (r-z) hough transforms to increase the efficiency rate. """
    # get the tracks run the pipeline to get the estimations of the phi-qpt transform
    all_xy_tracks = list(truth_df['xy_track'])
    _, xy_est_tracks_to_hits = xy_pipeline(all_xy_tracks, xy_hyperparams)
    
    # do the same for the r-z transform
    all_rz_tracks = list(truth_df['rz_track'])
    rz_intersections = compute_all_intersections(all_rz_tracks)
    rz_est_tracks_to_hits = rz_pipeline(all_rz_tracks, rz_intersections, rz_hyperparams)

    # get the union dictionaries
    xy_est_tracks_to_hits, rz_est_tracks_to_hits = unite_estimations(truth_df, xy_est_tracks_to_hits, rz_est_tracks_to_hits)

    # compute the rates for every "improved" transform and then return them
    truth_df['track'] = truth_df['xy_track']
    xy_eff = efficiency_rate(xy_est_tracks_to_hits.values(), truth_df)
    xy_fake = fake_rate(xy_est_tracks_to_hits.values(), truth_df)
    xy_dup = duplicate_rate(xy_est_tracks_to_hits.values(), truth_df)

    truth_df['track'] = truth_df['rz_track']
    rz_eff = efficiency_rate(rz_est_tracks_to_hits.values(), truth_df)
    rz_fake = fake_rate(rz_est_tracks_to_hits.values(), truth_df)
    rz_dup = duplicate_rate(rz_est_tracks_to_hits.values(), truth_df)
    
    return (xy_eff, xy_fake, xy_dup), (rz_eff, rz_fake, rz_dup)