import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm


def compute_qpt(xs, r, phi, hyperparams):
    """ Computes the q/p_T value of the Hough space from phi, r. """
    A = hyperparams['A']
    use_precise = hyperparams['use-precise-transform']
    return np.sin(xs - phi) / (A * r) if use_precise else (xs - phi) / (A * r)


def compute_b(xs, r, z, hyperparams):
    """ Computes the intercept `b` value of the Hough space for a line defined by slope z and intercept z. """
    return r * xs + z
    

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


def bins_with_least_hits(tracks_per_bin, min_hits):
    """ Returns a list of bins (tuple of ints) that have a minimum value. """
    return list(filter(lambda key: len(tracks_per_bin[key]) >= min_hits, tracks_per_bin))


def compute_track_to_hits_mapping(tracks_per_bin, top_indices, lower_phi, lower_qpt, bin_size):
    """ Creates a dictionary that maps: (est_phi, est_qpt) -> [hits they "cover"] """
    est_tracks_to_hits = {}
    for bin_x, bin_y in top_indices:
        key_x = lower_phi + bin_x * bin_size[0]
        key_y = lower_qpt + bin_y * bin_size[1]
        est_tracks_to_hits[(key_x, key_y)] = tracks_per_bin[(bin_x, bin_y)]
    return est_tracks_to_hits


def Hough2d_pipeline(tracks, hyperparams, compute_ys_func):
    """ Computes the 2D Hough transformation for a given function (compute_ys_func) and the tracks provided. """
    # unpack hyperparameters
    xrange, yrange = hyperparams['xrange'], hyperparams['yrange']
    bin_size = hyperparams['bin-size']
    nhits = hyperparams['minimum-hits-per-bin']

    # range of search in the x-axis values and their respective bins
    xs = np.arange(xrange[0], xrange[1], bin_size[0])
    xbins = int((xrange[1] - xrange[0]) / bin_size[0])
    ybins = int((yrange[1] - yrange[0]) / bin_size[1])

    # define the accumulator array and flatten it
    accumulator = np.zeros((xbins, ybins))
    flat_accumulator = np.ravel(accumulator)
    
    # vectorize the function for finding bins
    vect_find_bin = np.vectorize(find_bin)

    # book-keeping
    tracks_per_bin = {}

    # run through all the tracks
    for track in tracks:
        
        # compute the y-values for the according transformation
        ys = compute_ys_func(xs, track[0], track[1], hyperparams)

        # find in which bins the points for each transform belong to and remove those out of bounds
        bin_xs, bin_ys = vect_find_bin(xs, ys, xrange[0], yrange[0], bin_size[0], bin_size[1])
        illegal_indices = np.where((bin_ys < 0) | (bin_ys >= ybins))
        bin_xs = np.delete(bin_xs, illegal_indices)
        bin_ys = np.delete(bin_ys, illegal_indices)

        # flatten the indices in order to update the flat accumulators (which will automatically update the original accumulators)
        flat_indices = np.ravel_multi_index((bin_xs, bin_ys), accumulator.shape)
        flat_accumulator[flat_indices] += 1
        
        # update the book-keeping structures
        update_tracks_per_bin(tracks_per_bin, track, bin_xs, bin_ys)

    # get the indices of the bins that have at least the minimum required hits
    top_indices = bins_with_least_hits(tracks_per_bin, nhits)

    # get the dictionaries that map a hyperparameter pair to the hits it contains (in the corresponding bin)
    est_tracks_to_hits = compute_track_to_hits_mapping(tracks_per_bin, top_indices, xrange[0], yrange[0], bin_size)

    return accumulator, est_tracks_to_hits


def remove_found_hits(truth_df, est_tracks_to_hits):
    """ Removes the hits that were identified by the hough transform algorithm """
    hits_found = set()
    df_copy = truth_df.copy()
    for hits in est_tracks_to_hits.values():
        for hit in hits:
            if hit not in hits_found:
                df_copy = df_copy[df_copy['track'] != hit]
                hits_found.add(hit)
        if df_copy.shape[0] == 0:
            break
    return df_copy


def Hough2d_combined_pipeline(transform1_est_tracks_to_hits, truth_df, hyperparams, second_transform_is_rz=True):
    """ Deletes the hits found from the first Hough Transform, runs the other
        Hough Transform on the remaining hits and aggregates the results. """
    # find which hits to remove according to which hits have already been identified
    transform1_type = 'xy_track' if second_transform_is_rz else 'rz_track'
    transform2_type = 'rz_track' if second_transform_is_rz else 'xy_track'
    truth_df['track'] = truth_df[transform1_type]
    new_df = remove_found_hits(truth_df, transform1_est_tracks_to_hits)

    # run the other transform on the remaining hits
    tracks = list(new_df[transform2_type])
    compute_ys = compute_b if second_transform_is_rz else compute_qpt
    _, transform2_est_tracks_to_hits = Hough2d_pipeline(tracks, hyperparams, compute_ys)

    # aggregate the results
    for track, hits in transform2_est_tracks_to_hits.items():
        transform1_est_tracks_to_hits[track] = []
        for hit in hits:
            other_transform_track = truth_df[truth_df[transform2_type] == hit][transform1_type].iloc[0]
            transform1_est_tracks_to_hits[track].append(other_transform_track)

    return transform1_est_tracks_to_hits


def get_roi_accumulator(accumulator, hyperparams, rois):
    """ Returns a sub-array of an accumulator, containing the bins defined by the region of interest ranges. """
    xrange, yrange = hyperparams['xrange'], hyperparams['yrange']
    bin_size = hyperparams['bin-size']

    # define the RoI accumulator
    lower_x, upper_x = rois[0]
    low_x_bin = int((lower_x - xrange[0] - 1e-12) / bin_size[0])
    high_x_bin = int((upper_x - xrange[0] - 1e-12) / bin_size[0])

    lower_y, upper_y = rois[1]
    low_y_bin = int((lower_y - yrange[0] - 1e-12) / bin_size[1])
    high_y_bin = int((upper_y - yrange[0] - 1e-12) /bin_size[1])

    return accumulator[low_x_bin:high_x_bin, low_y_bin:high_y_bin]


def plot_heatmap(roi_accumulator, rois):
    """ Plots the number-of-hits-per-bin heatmap of the accumulator array. """
    # unpack data
    roi_xs, roi_ys = rois

    # define the figure and the ax
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))

    # x-axis ticks and labels
    nbins_x = 10
    ax.locator_params(axis='x', nbins=nbins_x)
    xticks = np.linspace(0, roi_accumulator.shape[0], nbins_x)
    x_range = np.linspace(roi_xs[0], roi_xs[1], xticks.shape[0])
    xtick_labels = ['{:.2f}'.format(tick) for tick in x_range]

    # y-axis ticks and labels
    nbins_y = 10
    ax.locator_params(axis='y', nbins=nbins_y)
    yticks = np.linspace(0, roi_accumulator.shape[1], nbins_y)
    y_range = np.linspace(roi_ys[0], roi_ys[1], yticks.shape[0])
    ytick_labels = ['{:.2f}'.format(tick) for tick in y_range]

    # heatmap
    ax = sns.heatmap(roi_accumulator.T, ax=ax)

    # config
    ax.set_xlabel('$\phi$', fontsize=15)
    ax.set_ylabel('$\\frac{q}{p_T}$', fontsize=15).set_rotation(0)

    ax.yaxis.set_label_coords(-0.1, 0.50); ax.invert_yaxis()
    ax.set_xticks(xticks); ax.set_xticklabels(xtick_labels, rotation=0)
    ax.set_yticks(yticks); ax.set_yticklabels(ytick_labels, rotation=0)
    ax.set_title('Number of tracks per bin.')

    plt.show()