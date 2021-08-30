import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils.hough_transform import compute_b, hough2d_pipeline


def get_roi_accumulator(accumulator, hyperparams, rois):
    """ Returns a sub-array of an accumulator, containing the bins
        defined by the region of interest ranges. """
    xrange, yrange = hyperparams['xrange'], hyperparams['yrange']
    bin_size = hyperparams['bin-size']

    # define the RoI accumulator
    lower_x, upper_x = rois[0]
    low_x_bin = int((lower_x - xrange[0] - 1e-12) / bin_size[0])
    high_x_bin = int((upper_x - xrange[0] - 1e-12) / bin_size[0])

    lower_y, upper_y = rois[1]
    low_y_bin = int((lower_y - yrange[0] - 1e-12) / bin_size[1])
    high_y_bin = int((upper_y - yrange[0] - 1e-12) / bin_size[1])

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
    ax.set_xlabel('$\\phi$', fontsize=15)
    ax.set_ylabel('$\\frac{q}{p_T}$', fontsize=15).set_rotation(0)

    ax.yaxis.set_label_coords(-0.1, 0.50)
    ax.invert_yaxis()
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=0)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, rotation=0)
    ax.set_title('Number of tracks per bin.')

    plt.show()


def tracks_are_close(track1, track2, thresholds):
    """ Returns true if the two given track are close,
        up to a threshold value. """
    return abs(track1[0] - track2[0]) <= thresholds[0] and abs(
        track1[1] - track2[1]) <= thresholds[1]


def common_hits_percentage(hits1, hits2):
    """ Returns the percentage of common hits for two estimated tracks. """
    hits1_set = set(hits1)
    hits2_set = set(hits2)
    return len(hits1_set & hits2_set) / len(hits1_set | hits2_set)


def get_track_to_geometry_mapping(truth_df):
    """ Returns a dictionary that maps {track -> geometry ID of that hit}. """
    return {series['track']: series['geometry_id']
            for _, series in truth_df.iterrows()}


def remove_hits_with_same_geometries(hits, hit_to_geom):
    """ Returns a list where there are no two hits with the same geometries. """
    geometry_to_count = {}
    for hit in hits:
        geom = hit_to_geom[hit]
        geometry_to_count[geom] = geometry_to_count.get(geom, 0) + 1

    new_hits = []
    for hit in hits:
        geom = hit_to_geom[hit]
        if geometry_to_count[geom] == 1:
            new_hits.append(hit)

    return new_hits


def filter_hits_with_same_geometries(est_tracks_to_hits, hit_to_geom):
    """ Returns a dictionary where the hits belonging to same geometry IDs
        have been removed. """
    filtered_est_tracks_to_hits = {}
    for key, hits in est_tracks_to_hits.items():
        new_hits = remove_hits_with_same_geometries(hits, hit_to_geom)
        if len(new_hits) > 0:
            filtered_est_tracks_to_hits[key] = new_hits
    return filtered_est_tracks_to_hits


def duplicate_removal_1(est_tracks_to_hits, closeness_thresholds,
                        similarity_threshold, remove_same_geometries=False,
                        hit_to_geom=None):
    """ Returns a clear-of-duplicates estimated_track to hits mapping. """
    # filter out hits (and maybe entire tracks) where their hits have same geoms
    if remove_same_geometries is True:
        est_tracks_to_hits = \
            filter_hits_with_same_geometries(est_tracks_to_hits, hit_to_geom)

    tracks = list(est_tracks_to_hits.keys())
    new_est_tracks_to_hits = {}
    duplicates = set()

    # scan all tracks
    for idx, t1 in enumerate(tracks):
        # if the current track has already been marked as a duplicate, proceed
        if t1 in duplicates:
            continue

        # else, scan all the tracks after it
        leading_track = t1
        for t2 in tracks[idx:]:

            h1, h2 = est_tracks_to_hits[leading_track], est_tracks_to_hits[t2]
            if tracks_are_close(leading_track, t2, closeness_thresholds) and \
                    common_hits_percentage(h1, h2) > similarity_threshold:

                # the remaining track becomes the one with the most hits
                duplicates.add(t2)
                if len(h1) < len(h2):
                    leading_track = t2

        # update the new estimations dictionary
        new_est_tracks_to_hits[leading_track] = \
            est_tracks_to_hits[leading_track]

    return new_est_tracks_to_hits


def duplicate_removal_2(est_tracks_to_hits, closeness_thresholds,
                        similarity_threshold, remove_same_geometries=False,
                        hit_to_geom=None):
    """ Returns a clear-of-duplicates estimated_track to hits mapping. """
    # filter out hits (and maybe entire tracks) where their hits have same geoms
    if remove_same_geometries is True:
        est_tracks_to_hits = \
            filter_hits_with_same_geometries(est_tracks_to_hits, hit_to_geom)

    tracks = list(est_tracks_to_hits.keys())
    new_est_tracks_to_hits = {}
    duplicates = set()

    # scan all tracks
    for idx, t1 in enumerate(tracks):
        # if the current track has already been marked as a duplicate, proceed
        if t1 in duplicates:
            continue

        # else, scan all the tracks after it
        leading_track = t1
        duplicates_for_current_track = {t1}
        for t2 in tracks[idx:]:
            # scan every duplicate track in the nearby region
            for dup_of_t1 in duplicates_for_current_track:

                h1, h2 = est_tracks_to_hits[dup_of_t1], est_tracks_to_hits[t2]
                if tracks_are_close(dup_of_t1, t2, closeness_thresholds) and \
                        common_hits_percentage(h1, h2) > similarity_threshold:

                    # the remaining track becomes the one with the most hits
                    duplicates.add(t2)
                    duplicates_for_current_track.add(t2)
                    if len(h1) < len(h2):
                        leading_track = t2
                    break

        # update the new estimations dictionary
        new_est_tracks_to_hits[leading_track] = \
            est_tracks_to_hits[leading_track]

    return new_est_tracks_to_hits


def convert_tracks(hits, truth_df, xy_to_rz=True):
    """ Given a list of xy tracks, it returns a list with the corresponding
        rz tracks. """
    first_transform, second_transform = ('xy_track', 'rz_track') if xy_to_rz \
        else ('rz_track', 'xy_track')
    return [series[second_transform] for _, series in truth_df.iterrows()
            if series[first_transform] in hits]


def plot_hits(hits, truth_df, xlims, xy=True):
    """ Given hits, it finds the corresponding hits of the other transformation
        and plots them. """
    xkey, ykey = ('tx', 'ty') if xy else ('tz', 'r')
    xlabel, ylabel = ('x', 'y') if xy else ('z', 'r')
    track = 'xy_track' if xy else 'rz_track'

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    for row, series in truth_df.iterrows():
        if series[track] in hits:
            ax.scatter(series[xkey], series[ykey])

    ax.set_xlabel(xlabel)
    ax.set_xlabel(ylabel)
    ax.set_xlim(xlims)
    ax.set_title(f'Hits in the {xlabel}-{ylabel} space')

    plt.show()


def purify_xy_hits(hits, rz_hyperparams, truth_df):
    """ Purify the hits-per-bin computed by the phi-q/p_T Hough Transform. """

    # run the r-z Hough Transform ONLY for the hits inside the `hits` list
    rz_tracks = convert_tracks(hits, truth_df, xy_to_rz=True)
    _, rz_est = hough2d_pipeline(rz_tracks, rz_hyperparams, compute_b)

    # pick the result with the highest number of hits in it
    best_track = max(rz_est, key=lambda track: len(rz_est[track])) \
        if len(rz_est) > 0 else None
    purified_rz_hits = rz_est[best_track] if best_track is not None else []

    # convert the hits in x-y tracks and return
    return convert_tracks(purified_rz_hits, truth_df, xy_to_rz=False)


def purify_xy_estimations(est_tracks_to_hits, rz_hyperparams, truth_df):
    """ Purifies the xy-estimated hits-per-bins by running the rz
        Hough Transform on the hits inside every bin (discretely). """
    purified_estimations = {}
    for est_track, hits in est_tracks_to_hits.items():
        purified_hits = purify_xy_hits(hits, rz_hyperparams, truth_df)
        if len(purified_hits) > 0:
            purified_estimations[est_track] = purified_hits
    return purified_estimations
