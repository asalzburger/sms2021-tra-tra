import os
import numpy as np
import pandas as pd
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


def run_pipeline_over_whole_datasets(hit_dfs, stats, xy_hyperparams, rz_hyperparams):
    """ Run all the pipelines over all the datasets. """

    def _update_stats(est_tracks_to_hits, truth_df, transform, _type):
        """ Updates the statistics dictionaries. """
        stats['efficiency'][transform][_type] += efficiency_rate(est_tracks_to_hits.values(), truth_df)
        stats['fake'][transform][_type] += fake_rate(est_tracks_to_hits.values(), truth_df)
        stats['duplicate'][transform][_type] += duplicate_rate(est_tracks_to_hits.values(), truth_df)

    def _run_pipelines(dfs, _type):
        """ Runs all the types of Hough Transforms for all the dataframes of a dataset. """

        desc = f'Running the Hough Transform for the simulation type: {_type}'
        for df in tqdm(dfs, desc=desc):

            # set appropriate values
            df['weight'] = 1.0
            df['r'] = np.sqrt(np.square(df['tx']) + np.square(df['ty']))
            df['phi'] = np.arctan2(df['ty'], df['tx'])
            df['xy_track'] = df[['r','phi']].apply(lambda pair: (pair[0], pair[1]), 1)
            df['rz_track'] = df[['r','tz']].apply(lambda pair: (-pair[0], pair[1]), 1)

            # compute the transforms on their own and update the statistics
            _, xy_est_tracks_to_hits = Hough2d_pipeline(list(df['xy_track']), xy_hyperparams, compute_qpt)
            _, rz_est_tracks_to_hits = Hough2d_pipeline(list(df['rz_track']), rz_hyperparams, compute_b)
            df['track'] = df['xy_track']; _update_stats(xy_est_tracks_to_hits, df, 'xy', _type)
            df['track'] = df['rz_track']; _update_stats(rz_est_tracks_to_hits, df, 'rz', _type)

            # compute the corresponding combinations and update the statistics
            xy_combo_rz_est_tracks_to_hits = Hough2d_combined_pipeline(xy_est_tracks_to_hits, df, rz_hyperparams, second_transform_is_rz=True)
            rz_combo_xy_est_tracks_to_hits = Hough2d_combined_pipeline(rz_est_tracks_to_hits, df, xy_hyperparams, second_transform_is_rz=False)
            df['track'] = df['xy_track']; _update_stats(xy_combo_rz_est_tracks_to_hits, df, 'xy-rz-combo', _type)
            df['track'] = df['rz_track']; _update_stats(rz_combo_xy_est_tracks_to_hits, df, 'rz-xy-combo', _type)
            
    for _type, type_hit_dfs in hit_dfs.items():
        _run_pipelines(type_hit_dfs, _type)


def populate_rates(rate_stats, simulations, transforms):
    """ Given a stats dictionary, returns a reshaped dict for every simulation type. """
    rate_dict = {}
    for simul in simulations:
        rate_dict[simul] = {}
        for transform in transforms:
            rate_dict[simul][transform] = rate_stats[transform][simul] / 100
    return rate_dict


def plot_rates(rates_dict, simulations, metric_type):
    """ Plots a specific {Metric_type} Rate vs Transform type plot for every simulation type. """
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    for simul in simulations:
        ax.plot(rates_dict[simul].keys(), rates_dict[simul].values(), label=simul)
    ax.set_xlabel('Transform Type')
    ax.set_ylabel(f'{metric_type} Rate\n')
    ax.set_title(f'{metric_type} Rate per Transform Type')
    ax.legend()
    plt.show()


def group_all_dfs_per_eta(hit_dfs, initial_dfs, dfs_per_eta, eta_min, eta_bin_length, eta_keys):
    """ Builds the `dfs_per_eta` dictionary so that it contains all the
        tracks with common eta values. """

    def _group_dfs_per_eta(simul_dfs, simul_initial_dfs, simul_dfs_per_eta, eta_min, eta_bin_length, eta_keys):
        """ Iterate through all the dataframes and start placing each
            particle track in the corresponding eta dataframe. """

        # place the truth tracks in the according dataframes
        for idx, (df, initial_df) in enumerate(zip(simul_dfs, simul_initial_dfs)):
            
            # add important values
            df['weight'] = 1
            df['r'] = np.sqrt(np.square(df['tx']) + np.square(df['ty']))
            df['phi'] = np.arctan2(df['ty'], df['tx'])
            df['xy_track'] = df[['r','phi']].apply(lambda pair: (pair[0], pair[1]), 1)
            df['rz_track'] = df[['r','tz']].apply(lambda pair: (-pair[0], pair[1]), 1)
        
            # compute eta
            initial_df['|p|'] = np.sqrt(initial_df['px'] ** 2 + initial_df['py'] ** 2 + initial_df['pz'] ** 2)
            initial_df['theta'] = np.arccos(initial_df['pz'] / initial_df['|p|'])
            initial_df['eta'] = -np.log(np.tan(initial_df['theta'] / 2))

            _df = df.copy()
            unique_particles = set(_df['particle_id'])
            _df['particle_id'] = _df['particle_id'] + '_{}'.format(idx)

            # for every unique particle find it's eta value and place the track to the correct df
            for pid in unique_particles:

                particle_df = _df[_df['particle_id'] == pid + '_{}'.format(idx)]
    
                eta = initial_df.loc[initial_df['particle_id'] == pid, 'eta'].item()
                _bin = int((eta - eta_min - 1e-12) / eta_bin_length)

                eta_range = eta_keys[_bin]
                simul_dfs_per_eta[eta_range] = pd.concat([simul_dfs_per_eta[eta_range], particle_df], sort=False)
                
        # reset the indices
        for eta_range in eta_keys:
            simul_dfs_per_eta[eta_range] = simul_dfs_per_eta[eta_range].reset_index().drop(['level_0'], axis=1)

    simulations = hit_dfs.keys()
    for simul in simulations:
        _group_dfs_per_eta(hit_dfs[simul], initial_dfs[simul], dfs_per_eta[simul], eta_min, eta_bin_length, eta_keys)



def group_all_dfs_per_pt(hit_dfs, initial_dfs, dfs_per_pt, pt_min, pt_bin_length, pt_keys):
    """ Builds the `dfs_per_pt` dictionary so that it contains all the
        tracks with common p_T values. """

    def _group_dfs_per_pt(simul_dfs, simul_initial_dfs, simul_dfs_per_pt, pt_min, pt_bin_length, pt_keys):
        """ Iterate through all the dataframes and start placing each
            particle track in the corresponding pt dataframe. """

        # place the truth tracks in the according dataframes
        for idx, (df, initial_df) in enumerate(zip(simul_dfs, simul_initial_dfs)):
            
            # add important values
            df['weight'] = 1
            df['r'] = np.sqrt(np.square(df['tx']) + np.square(df['ty']))
            df['phi'] = np.arctan2(df['ty'], df['tx'])
            df['xy_track'] = df[['r','phi']].apply(lambda pair: (pair[0], pair[1]), 1)
            df['rz_track'] = df[['r','tz']].apply(lambda pair: (-pair[0], pair[1]), 1)
        
            # compute pt
            initial_df['pt'] = np.sqrt(initial_df['px'] ** 2 + initial_df['py'] ** 2)

            _df = df.copy()
            unique_particles = set(_df['particle_id'])
            _df['particle_id'] = _df['particle_id'] + '_{}'.format(idx)

            # for every unique particle find it's pt value and place the track to the correct df
            for pid in unique_particles:

                particle_df = _df[_df['particle_id'] == pid + '_{}'.format(idx)]
    
                pt = initial_df.loc[initial_df['particle_id'] == pid, 'pt'].item()
                _bin = int((pt - pt_min - 1e-12) / pt_bin_length)

                pt_range = pt_keys[_bin]
                simul_dfs_per_pt[pt_range] = pd.concat([simul_dfs_per_pt[pt_range], particle_df], sort=False)
                
        # reset the indices
        for pt_range in pt_keys:
            simul_dfs_per_pt[pt_range] = simul_dfs_per_pt[pt_range].reset_index().drop(['level_0'], axis=1)

    simulations = hit_dfs.keys()
    for simul in simulations:
        _group_dfs_per_pt(hit_dfs[simul], initial_dfs[simul], dfs_per_pt[simul], pt_min, pt_bin_length, pt_keys)


def compute_eta_stats(eta_stats, dfs_per_eta, xy_hyperparams, rz_hyperparams, eta_keys):
    """ Runs the combined pipeline for all eta values and updates the stats dictionary. """

    def _update_stats(est_tracks_to_hits, truth_df, simul, range):
        """ Updates the statistics dictionaries. """
        eta_stats['efficiency'][simul][range] += efficiency_rate(est_tracks_to_hits.values(), truth_df)
        eta_stats['fake'][simul][range] += fake_rate(est_tracks_to_hits.values(), truth_df)
        eta_stats['duplicate'][simul][range] += duplicate_rate(est_tracks_to_hits.values(), truth_df)
    
    def _compute_eta_for_simul(simul_dfs_per_eta, simul):
        """ Runs the combined pipeline for 1 simulation. """
        # for all eta ranges
        desc = f'Computing eta metrics for simulation: {simul}'
        for eta_range in tqdm(eta_keys, desc=desc):
            # get the dataframe, run the Hough Transform and update the stats
            df = simul_dfs_per_eta[eta_range]
            _, xy_est = Hough2d_pipeline(list(df['xy_track']), xy_hyperparams, compute_qpt)
            xy_combo_rz_est = Hough2d_combined_pipeline(xy_est, df, rz_hyperparams, second_transform_is_rz=True)
            _update_stats(xy_combo_rz_est, df, simul, eta_range)

    simulations = dfs_per_eta.keys()
    for simul in simulations:
        _compute_eta_for_simul(dfs_per_eta[simul], simul)


def compute_pt_stats(pt_stats, dfs_per_pt, xy_hyperparams, rz_hyperparams, pt_keys):
    """ Runs the combined pipeline for all pt values and updates the stats dictionary. """

    def _update_stats(est_tracks_to_hits, truth_df, simul, range):
        """ Updates the statistics dictionaries. """
        pt_stats['efficiency'][simul][range] += efficiency_rate(est_tracks_to_hits.values(), truth_df)
        pt_stats['fake'][simul][range] += fake_rate(est_tracks_to_hits.values(), truth_df)
        pt_stats['duplicate'][simul][range] += duplicate_rate(est_tracks_to_hits.values(), truth_df)
    
    def _compute_pt_for_simul(simul_dfs_per_pt, simul):
        """ Runs the combined pipeline for 1 simulation. """
        # for all pt ranges
        desc = f'Computing pt metrics for simulation: {simul}'
        for pt_range in tqdm(pt_keys, desc=desc):
            # get the dataframe, run the Hough Transform and update the stats
            df = simul_dfs_per_pt[pt_range]
            _, xy_est = Hough2d_pipeline(list(df['xy_track']), xy_hyperparams, compute_qpt)
            xy_combo_rz_est = Hough2d_combined_pipeline(xy_est, df, rz_hyperparams, second_transform_is_rz=True)
            _update_stats(xy_combo_rz_est, df, simul, pt_range)

    simulations = dfs_per_pt.keys()
    for simul in simulations:
        _compute_pt_for_simul(dfs_per_pt[simul], simul)


def plot_eta_stats(eta_stats, eta_keys):
    """ Plots metric statistics per eta range per simulation """
    eval_metrics = ['efficiency', 'fake', 'duplicate']
    xticks = np.array(range(len(eta_keys)))
    vals = lambda d: [d[eta_keys[x]] for x in xticks] 
    
    fig, axes = plt.subplots(3, 1, figsize=(30, 40))
    
    for metric, ax in zip(eval_metrics, axes):
    
        metric_dict = eta_stats[metric]
        for transform, stats in metric_dict.items():
            ax.plot(xticks, vals(stats), label=transform)

        ax.set_xticks(xticks)
        ax.set_xticklabels(eta_keys, fontsize=16)
        ax.set_xlabel('$\eta$ range', fontsize=20)
        ax.set_ylabel('{} rate\n'.format(metric), fontsize=20)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(16) 
        ax.set_title('{} rate vs $\eta$ range'.format(metric), fontsize=20)
        ax.legend()
    
    plt.show()


def plot_pt_stats(pt_stats, pt_keys):
    """ Plots metric statistics per pt range per simulation """
    eval_metrics = ['efficiency', 'fake', 'duplicate']
    xticks = np.array(range(len(pt_keys)))
    vals = lambda d: [d[pt_keys[x]] for x in xticks] 
    
    fig, axes = plt.subplots(3, 1, figsize=(30, 40))
    
    for metric, ax in zip(eval_metrics, axes):
    
        metric_dict = pt_stats[metric]
        for transform, stats in metric_dict.items():
            ax.plot(xticks, vals(stats), label=transform)

        ax.set_xticks(xticks)
        ax.set_xticklabels(pt_keys, fontsize=16)
        ax.set_xlabel('$p_T$ range', fontsize=20)
        ax.set_ylabel('{} rate\n'.format(metric), fontsize=20)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(16) 
        ax.set_title('{} rate vs $p_T$ range'.format(metric), fontsize=20)
        ax.legend()
    
    plt.show()
