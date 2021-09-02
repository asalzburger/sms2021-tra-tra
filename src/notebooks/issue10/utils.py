import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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
