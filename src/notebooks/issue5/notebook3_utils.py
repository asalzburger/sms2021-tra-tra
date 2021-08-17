import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_tracks_in_hough_space(tracks, phi_range, ylims, A, colors):
    """ Plots the tracks in the parameter space. """
    # define the figure and the axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

    # start plotting
    x_range = np.arange(phi_range[0], phi_range[1], 0.01)
    for idx, track in enumerate(tracks):
        r, phi = track
        precise_ys = np.sin(x_range - phi) / (A * r)
        approx_ys = (x_range - phi) / (A * r)
        ax1.plot(x_range, precise_ys, color=colors[idx % len(colors)])
        ax2.plot(x_range, approx_ys, color=colors[idx % len(colors)])

    ax1.set_xlabel('$\phi_0$', fontsize=15)
    h = ax1.set_ylabel('$\\frac{q}{p_T}$', fontsize=15)
    h.set_rotation(0)
    ax1.set_ylim(ylims[0], ylims[1])
    ax1.set_title('Hough Space for Precise Transformation')

    ax2.set_xlabel('$\phi_0$', fontsize=15)
    h = ax2.set_ylabel('$\\frac{q}{p_T}$', fontsize=15)
    h.set_rotation(0)
    ax2.set_ylim(ylims[0], ylims[1])
    ax2.set_title('Hough Space for Approximated Transformation')

    plt.show()


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


def populate_accumulators(tracks, precise_hyperparams, approx_hyperparams):
    """ Creates 2 accumulator arrays and populates them with values from
        Precise and Approximated Hough Transforms respectively. """

    # unpack hyperparameters and constants
    precise_phi_range, precise_qpt_range = precise_hyperparams['phi-range'], precise_hyperparams['qpt-range']
    precise_bin_size = precise_hyperparams['bin-size']
    precise_nhits = precise_hyperparams['minimum-number-of-hits']
    precise_A = precise_hyperparams['A']

    approx_phi_range, approx_qpt_range = approx_hyperparams['phi-range'], approx_hyperparams['qpt-range']
    approx_bin_size = approx_hyperparams['bin-size']
    approx_nhits = approx_hyperparams['minimum-number-of-hits']
    approx_A = approx_hyperparams['A']

    # ranges of search
    precise_xs = np.arange(precise_phi_range[0], precise_phi_range[1], precise_bin_size[0])
    precise_xbins = int((precise_phi_range[1] - precise_phi_range[0]) / precise_bin_size[0])
    precise_ybins = int((precise_qpt_range[1] - precise_qpt_range[0]) / precise_bin_size[1])

    approx_xs = np.arange(approx_phi_range[0], approx_phi_range[1], approx_bin_size[0])
    approx_xbins = int((approx_phi_range[1] - approx_phi_range[0]) / approx_bin_size[0])
    approx_ybins = int((approx_qpt_range[1] - approx_qpt_range[0]) / approx_bin_size[1])


    # define the accumulator arrays and flatten them
    precise_accumulator = np.zeros((precise_xbins, precise_ybins))
    flat_precise_accumulator = np.ravel(precise_accumulator)
    
    approx_accumulator = np.zeros((approx_xbins, approx_ybins))
    flat_approx_accumulator = np.ravel(approx_accumulator)

    # vectorize the function of finding bins
    vect_find_bin = np.vectorize(find_bin)

    # book-keeping
    tracks_per_precise_bin = {}
    tracks_per_approx_bin = {}

    # run through all the tracks
    for track in tracks:
        r, phi = track
        precise_ys = np.sin(precise_xs - phi) / (precise_A * r)
        approx_ys = (approx_xs - phi) / (approx_A * r)

        # find in which bins the points for each transform belong to
        precise_bin_xs, precise_bin_ys = vect_find_bin(precise_xs, precise_ys, precise_phi_range[0], precise_qpt_range[0],
                                                       precise_bin_size[0], precise_bin_size[1])
        approx_bin_xs, approx_bin_ys = vect_find_bin(approx_xs, approx_ys, approx_phi_range[0], approx_qpt_range[0],
                                                     approx_bin_size[0], approx_bin_size[1])
        
        # flatten the indices in order to update the flat accumulators (which will automatically update the original accumulators)
        flat_precise_indices = np.ravel_multi_index((precise_bin_xs, precise_bin_ys), precise_accumulator.shape)
        flat_precise_accumulator[flat_precise_indices] += 1
        
        flat_approx_indices = np.ravel_multi_index((approx_bin_xs, approx_bin_ys), approx_accumulator.shape)
        flat_approx_accumulator[flat_approx_indices] += 1

        # update the book-keeping structures
        update_tracks_per_bin(tracks_per_precise_bin, track, precise_bin_xs, precise_bin_ys)
        update_tracks_per_bin(tracks_per_approx_bin, track, approx_bin_xs, approx_bin_ys)

    # get the indices of the bins that have at least the minimum required hits
    top_indices_precise = bins_with_least_hits(tracks_per_precise_bin, precise_nhits)
    top_indices_approx = bins_with_least_hits(tracks_per_approx_bin, approx_nhits)

    # get the dictionaries that map a hyperparameter pair to the hits it contains (in the corresponding bin)
    precise_est_tracks_to_hits = compute_track_to_hits_mapping(tracks_per_precise_bin, top_indices_precise,
                                                               precise_phi_range[0], precise_qpt_range[0], precise_bin_size)
    approx_est_tracks_to_hits = compute_track_to_hits_mapping(tracks_per_approx_bin, top_indices_approx,
                                                              approx_phi_range[0], approx_qpt_range[0], approx_bin_size)

    return (precise_accumulator, precise_est_tracks_to_hits), (approx_accumulator, approx_est_tracks_to_hits)


def get_roi_accumulators(accumulators, precise_hyperparams, approx_hyperparams, precise_rois, approx_rois):
    """ Returns a sub-array for each accumulator, containing the bins defined by the region of interest ranges. """
    precise_accumulator, approx_accumulator = accumulators

    precise_phi_range, precise_qpt_range = precise_hyperparams['phi-range'], precise_hyperparams['qpt-range']
    approx_phi_range, approx_qpt_range = approx_hyperparams['phi-range'], approx_hyperparams['qpt-range']

    precise_bin_size = precise_hyperparams['bin-size']
    approx_bin_size = approx_hyperparams['bin-size']

    # define the RoI accumulator for the precise transformation
    precise_lower_phi, precise_upper_phi = precise_rois[0]
    precise_low_phi_bin = int((precise_lower_phi - precise_phi_range[0] - 1e-12) / precise_bin_size[0])
    precise_high_phi_bin = int((precise_upper_phi - precise_phi_range[0] - 1e-12) / precise_bin_size[0])

    precise_lower_qpt, precise_upper_qpt = precise_rois[1]
    precise_low_qpt_bin = int((precise_lower_qpt - precise_qpt_range[0] - 1e-12) / precise_bin_size[1])
    precise_high_qpt_bin = int((precise_upper_qpt - precise_qpt_range[0] - 1e-12) / precise_bin_size[1])

    roi_precise_accumulator = precise_accumulator[precise_low_phi_bin:precise_high_phi_bin, precise_low_qpt_bin:precise_high_qpt_bin]

    # define the RoI accumulator for the approximated transformation
    approx_lower_phi, approx_upper_phi = approx_rois[0]
    approx_low_phi_bin = int((approx_lower_phi - approx_phi_range[0] - 1e-12) / approx_bin_size[0])
    approx_high_phi_bin = int((approx_upper_phi - approx_phi_range[0] - 1e-12) / approx_bin_size[0])

    approx_lower_qpt, approx_upper_qpt = approx_rois[1]
    approx_low_qpt_bin = int((approx_lower_qpt - approx_qpt_range[0] - 1e-12) / approx_bin_size[1])
    approx_high_qpt_bin = int((approx_upper_qpt - approx_qpt_range[0] - 1e-12) / approx_bin_size[1])

    roi_approx_accumulator = approx_accumulator[approx_low_phi_bin:approx_high_phi_bin, approx_low_qpt_bin:approx_high_qpt_bin]

    return roi_precise_accumulator, roi_approx_accumulator


def plot_heatmaps(accumulators, precise_rois, approx_rois):
    """ Plots 2 heatmaps: 1 for the precise and 1 for the apprximated transforms, respectively. """
    # unpack data
    roi_precise_accumulator, roi_approx_accumulator = accumulators
    precise_roi_phis, precise_roi_qpts = precise_rois
    approx_roi_phis, approx_roi_qpts = approx_rois

    # define the figure and the axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

    # x-axis ticks and labels for both plots
    nbins_x = 10
    ax1.locator_params(axis='x', nbins=nbins_x)
    precise_xticks = np.linspace(0, roi_precise_accumulator.shape[0], nbins_x)
    precise_x_range = np.linspace(precise_roi_phis[0], precise_roi_phis[1], precise_xticks.shape[0])
    precise_xtick_labels = ['{:.2f}'.format(tick) for tick in precise_x_range]

    nbins_x = 10
    ax2.locator_params(axis='x', nbins=nbins_x)
    approx_xticks = np.linspace(0, roi_approx_accumulator.shape[0], nbins_x)
    approx_x_range = np.linspace(approx_roi_phis[0], approx_roi_phis[1], approx_xticks.shape[0])
    approx_xtick_labels = ['{:.2f}'.format(tick) for tick in approx_x_range]

    # y-axis ticks and labels for both plots
    nbins_y = 10
    ax1.locator_params(axis='y', nbins=nbins_y)
    precise_yticks = np.linspace(0, roi_precise_accumulator.shape[1], nbins_y)
    precise_y_range = np.linspace(precise_roi_qpts[0], precise_roi_qpts[1], precise_yticks.shape[0])
    precise_ytick_labels = ['{:.2f}'.format(tick) for tick in precise_y_range]

    nbins_y = 10
    ax2.locator_params(axis='y', nbins=nbins_y)
    approx_yticks = np.linspace(0, roi_approx_accumulator.shape[1], nbins_y)
    approx_y_range = np.linspace(approx_roi_qpts[0], approx_roi_qpts[1], approx_yticks.shape[0])
    approx_ytick_labels = ['{:.2f}'.format(tick) for tick in approx_y_range]

    # heatmaps
    ax1 = sns.heatmap(roi_precise_accumulator.T, ax=ax1)
    ax2 = sns.heatmap(roi_approx_accumulator.T, ax=ax2)

    # config
    ax1.set_xlabel('$\phi$', fontsize=15)
    ax2.set_xlabel('$\phi$', fontsize=15)
    h = ax1.set_ylabel('$\\frac{q}{p_T}$', fontsize=15); h.set_rotation(0)
    h = ax2.set_ylabel('$\\frac{q}{p_T}$', fontsize=15); h.set_rotation(0)

    ax1.yaxis.set_label_coords(-0.15, 0.50); ax1.invert_yaxis()
    ax2.yaxis.set_label_coords(-0.15, 0.50); ax2.invert_yaxis()

    ax1.set_xticks(precise_xticks); ax1.set_xticklabels(precise_xtick_labels, rotation=0)
    ax2.set_xticks(approx_xticks); ax2.set_xticklabels(approx_xtick_labels, rotation=0)

    ax1.set_yticks(precise_yticks); ax1.set_yticklabels(precise_ytick_labels, rotation=0)
    ax2.set_yticks(approx_yticks); ax2.set_yticklabels(approx_ytick_labels, rotation=0)

    ax1.set_title('Precise Transformation: Number of tracks per bin.')
    ax2.set_title('Approximated Transformation: Number of tracks per bin.')

    plt.show()


def get_str(x, n=2):
    """ returns a float as a string and cuts at the n-th decimal digit """
    x = truncate(x, n)
    x = 0.0 if x == 0.0 else x
    before, after = str(x).split('.', 1)
    return '.'.join([before, after[:n]])


def truncate(f, n):
    """ Truncates the digits of a floating point number. e.g.: f(0.8756343, 3) = 0.875 """
    return math.floor(f * 10 ** n) / 10 ** n