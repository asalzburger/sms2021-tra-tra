import os
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.abspath(os.path.join('..', '..', 'utils')))

from tqdm.notebook import tqdm
from metrics import efficiency_rate, fake_rate, duplicate_rate



def plot_views(dfs, types, x_field, y_field, x_label, y_label, colors):
    """ Plots the views (for the fields provided) for each one of the
        data frames provided (ideal, mat, non-homogen, mat-non-homogen). """

    def plot_view(ax, df, _type):
        """ Plots the hit view of a given data frame in a specific `Axes` object. """
        
        particles = list(df['particle_id'])
        for idx, pid in enumerate(particles):
            particle_df = df[df['particle_id'] == pid]
            ax.scatter(x=particle_df[x_field], y=particle_df[y_field], color=colors[idx % len(colors)])
            ax.set_xlabel(x_field)
            ax.set_ylabel(y_field)
            title = f'\nMulti-particle trajectories in the {x_label}-{y_label} plane for the {_type} simulation.\n'
            ax.set_title(title)

    fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2, figsize=(20, 10))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=1.1, wspace=None, hspace=0.4)

    plot_view(ax11, dfs[0], types[0])
    plot_view(ax12, dfs[1], types[1])
    plot_view(ax21, dfs[2], types[2])
    plot_view(ax22, dfs[3], types[3])

    plt.show()


def plot_xy_hough(dfs, A, types, phi_range, ylims, use_precise_transform=False):
    """ Plots the tracks in the phi-qp_T parameter space. """
    fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2, figsize=(20, 10))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=1.1, wspace=None, hspace=0.4)

    def _plot_xy(ax, df, _type):
        """ Plots the tracks of a given dataframe in a specified axis. """
        xs = np.arange(phi_range[0], phi_range[1], 0.01)

        tracks = list(df['xy_track'])
        for track in tracks:
            r, phi = track
            ys = np.sin(xs - phi) / (A * r) if use_precise_transform else (xs - phi) / (A * r)
            ax.plot(xs, ys)
            
        ax.set_ylim(ylims[0], ylims[1])
        suffix = '\n' + ('Precise' if use_precise_transform else 'Approximated') + ' Transform '
        title = suffix + '$phi$ - $\\frac{q}{p_T}$ Hough space for the ' + '{} simulation.\n'.format(_type)
        ax.set_title(title)
        ax.set_xlabel('$\phi_0$', fontsize=15)
        ax.set_ylabel('$\\frac{q}{p_T}$', fontsize=15).set_rotation(0)

    _plot_xy(ax11, dfs[0], types[0])
    _plot_xy(ax12, dfs[1], types[1])
    _plot_xy(ax21, dfs[2], types[2])
    _plot_xy(ax22, dfs[3], types[3])
            
    plt.show()


def plot_rz_hough(dfs, types, phi_range, ylims, use_precise_transform=False):
    """ Plots the tracks in the r-z parameter space. """
    fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2, figsize=(20, 10))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=1.1, wspace=None, hspace=0.4)

    def _plot_rz(ax, df, _type):
        """ Plots the tracks of a given dataframe in a specified axis. """
        xs = np.arange(phi_range[0], phi_range[1], 0.01)

        tracks = list(df['rz_track'])
        for track in tracks:
            m, b = track
            ys = m * xs + b
            ax.plot(xs, ys)
            
        ax.set_ylim(ylims[0], ylims[1])
        title = '\n$r$ - $z$ Hough space for the ' + '{} simulation.\n'.format(_type)
        ax.set_title(title)
        ax.set_xlabel('$r$', fontsize=15)
        ax.set_ylabel('$z$', fontsize=15).set_rotation(0)

    _plot_rz(ax11, dfs[0], types[0])
    _plot_rz(ax12, dfs[1], types[1])
    _plot_rz(ax21, dfs[2], types[2])
    _plot_rz(ax22, dfs[3], types[3])
            
    plt.show()


def compute_qpt(xs, r, phi, hyperparams):
    """ Computes the q/p_T value of the Hough space from phi, r. """
    A = hyperparams['A']
    use_precise = hyperparams['use-precise-transform']
    return np.sin(xs - phi) / (A * r) if use_precise else (xs - phi) / (A * r)


def compute_b(xs, r, z, hyperparams):
    """ Computes the intercept `b` value of the Hough space for a line defined by slope z and intercept z. """
    return xs * r + z
    

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

        # find in which bins the points for each transform belong to
        bin_xs, bin_ys = vect_find_bin(xs, ys, xrange[0], yrange[0], bin_size[0], bin_size[1])

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
    df_copy = truth_df.copy()
    for track, hits in est_tracks_to_hits.items():
        for hit in hits:
            df_copy = df_copy[df_copy['track'] != hit]
    return df_copy


def Hough2d_combined_pipeline(transform1_est_tracks_to_hits, truth_df, hyperparams, use_rz=True):
    """ Deletes the hits found from the first Hough Transform and runs the other Hough Transform on the remaining hits. """
    # find which hits to remove according to which hits have already been identified
    truth_df['track'] = truth_df['xy_track'] if use_rz else truth_df['rz_track']
    new_df = remove_found_hits(truth_df, transform1_est_tracks_to_hits)

    # run the other transform on the remaining hits
    tracks = list(new_df['rz_track']) if use_rz else list(new_df['xy_track'])
    compute_ys = compute_b if use_rz else compute_qpt
    return Hough2d_pipeline(tracks, hyperparams, compute_ys)


def run_pipeline_over_whole_datasets(hit_dfs, stats, xy_hyperparams, rz_hyperparams):
    """  """

    def _update_stats(est_tracks_to_hits, truth_df, transform, _type):
        """ Updates the statistics dictionaries. """
        stats['efficiency'][transform][_type] += efficiency_rate(est_tracks_to_hits.values(), truth_df)
        stats['fake'][transform][_type] += fake_rate(est_tracks_to_hits.values(), truth_df)
        stats['duplicate'][transform][_type] += duplicate_rate(est_tracks_to_hits.values(), truth_df)

    def _run_pipelines(dfs, _type):
        """ Runs all the types of Hough Transforms for all the dataframes of a dataset. """

        desc = f'Running the Hough Transform for the simulation type: {_type}'
        for df in tqdm(dfs, desc=desc):

            print('peoss\n')

            # set appropriate values
            df['weight'] = 1.0
            df['r'] = np.sqrt(np.square(df['tx']) + np.square(df['ty']))
            df['phi'] = np.arctan2(df['ty'], df['tx'])
            df['xy_track'] = df[['r','phi']].apply(lambda pair: (pair[0], pair[1]), 1)
            df['rz_track'] = df[['r','tz']].apply(lambda pair: (-pair[0], pair[1]), 1)

            # compute the transforms on their own
            _, xy_est_tracks_to_hits = Hough2d_pipeline(list(df['xy_track']), xy_hyperparams, compute_qpt)
            _, rz_est_tracks_to_hits = Hough2d_pipeline(list(df['rz_track']), rz_hyperparams, compute_b)

            # compute the corresponding combinations
            _, xy_combo_rz_est_tracks_to_hits = Hough2d_combined_pipeline(xy_est_tracks_to_hits, df, rz_hyperparams, use_rz=True)
            _, rz_combo_xy_est_tracks_to_hits = Hough2d_combined_pipeline(rz_est_tracks_to_hits, df, xy_hyperparams, use_rz=False)

            # update the statistics
            df['track'] = df['xy_track']
            _update_stats(xy_est_tracks_to_hits, df, 'xy', _type)
            _update_stats(rz_combo_xy_est_tracks_to_hits, df, 'rz-xy-combo', _type)

            df['track'] = df['rz_track']
            print()
            print(rz_est_tracks_to_hits)
            print()
            _update_stats(rz_est_tracks_to_hits, df, 'rz', _type)
            _update_stats(xy_combo_rz_est_tracks_to_hits, df, 'xy-rz-combo', _type)

    for _type, type_hit_dfs in hit_dfs.items():
        _run_pipelines(type_hit_dfs, _type)
                


def get_str(x, n=2):
    """ returns a float as a string and cuts at the n-th decimal digit """
    x = truncate(x, n)
    x = 0.0 if x == 0.0 else x
    before, after = str(x).split('.', 1)
    return '.'.join([before, after[:n]])


def truncate(f, n):
    """ Truncates the digits of a floating point number. e.g.: f(0.8756343, 3) = 0.875 """
    return math.floor(f * 10 ** n) / 10 ** n