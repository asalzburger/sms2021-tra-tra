import numpy as np
import matplotlib.pyplot as plt

import src.utils.metrics as metrics


def get_range_labels(ranges, n=2):
    """ Convert range labels to floats with at most `n` decimal digits. """
    return ['{} – {}'.format(round(ranges[i], n), round(ranges[i + 1], n))
            for i in range(len(ranges) - 1)]


def compute_counts(values, nbins, max_value=1.0):
    """ Returns an array where every cell is the count of values falling in
        the bins it corresponds to. Basically it digitizes `values`. """
    counts = np.zeros(nbins)
    for v in values:
        _bin = int(v * nbins) if v < max_value else nbins - 1
        counts[_bin] += 1
    return counts


def plot_matching_probability_vs_count(est_tracks_to_hits, truth_df):
    """ Plots matching-probability vs count for all the estimated tracks. """
    nbins = 10
    mp_ranges = np.linspace(0, 1, nbins + 1)
    mp_ranges_str = get_range_labels(mp_ranges, n=2)

    track_to_truth_row = metrics.get_track_to_truth_row_mapping(truth_df)
    mps = [metrics.matching_probability(hits, track_to_truth_row, truth_df)[1]
           for hits in est_tracks_to_hits.values()]
    counts = compute_counts(mps, nbins)

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    ax.plot(mp_ranges_str, counts, color='red')
    ax.set_xlabel('Matching Probability')
    ax.set_ylabel('Count')
    ax.set_title('Count vs Matching-Probability range')

    plt.show()


def plot_purity_vs_count(est_tracks_to_hits, truth_df):
    """ Plots purity vs count for all the estimated tracks. """
    nbins = 10
    purity_ranges = np.linspace(0, 1, nbins + 1)
    purity_ranges_str = get_range_labels(purity_ranges, n=2)

    track_to_truth_row = metrics.get_track_to_truth_row_mapping(truth_df)
    purities = [metrics.purity(hits, track_to_truth_row, truth_df)[1]
                for hits in est_tracks_to_hits.values()]
    counts = compute_counts(purities, nbins)

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    ax.plot(purity_ranges_str, counts, color='navy')
    ax.set_xlabel('Purity')
    ax.set_ylabel('Count')
    ax.set_title('Count vs Purity range')

    plt.show()
