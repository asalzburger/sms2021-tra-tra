import numpy as np


def preprocess_hit_df(df):
    """ Preprocesses a `hits` dataframe by adding the relevant columns. """
    df['weight'] = 1
    df['r'] = np.sqrt(np.square(df['tx']) + np.square(df['ty']))
    df['phi'] = np.arctan2(df['ty'], df['tx'])
    df['xy_track'] = df[['r', 'phi']].apply(lambda pair: (pair[0], pair[1]), 1)
    df['rz_track'] = df[['r', 'tz']].apply(lambda pair: (-pair[0], pair[1]), 1)


def preprocess_initial_final_df(df):
    """ Preprocesses a `initial-` or `final-` dataframe by adding the
        relevant columns. """
    df['pt'] = np.sqrt(df['px'] ** 2 + df['py'] ** 2)
    df['|p|'] = np.sqrt(df['px'] ** 2 + df['py'] ** 2 + df['pz'] ** 2)
    df['theta'] = np.arccos(df['pz'] / df['|p|'])
    df['eta'] = -np.log(np.tan(df['theta'] / 2))


def get_list_based_data(df):
    """ Given a pandas DataFrame, it returns in lists the useful data, so that
        they can be accessed in O(1) time. """
    xy_tracks = list(df['xy_track'])
    rz_tracks = list(df['rz_track'])
    particles = list(set(df['particle_id']))
    pid_mapping = {pid: index for index, pid in enumerate(particles)}
    pids = [pid_mapping[pid] for pid in list(df['particle_id'])]
    weights = list(df['weight'])
    pid_to_total_weight = [0.0] * len(particles)
    for pid_index, weight in zip(pids, weights):
        pid_to_total_weight[pid_index] += weight

    return xy_tracks, rz_tracks, particles, pids, weights, pid_to_total_weight


def get_track_to_truth_row_mapping(truth_df):
    """ Returns a dictionary that maps a track to its
        row index in the dataframe. """
    return {series['track']: row for row, series in truth_df.iterrows()}


def get_track_to_geometry_mapping(truth_df):
    """ Returns a dictionary that maps {track -> geometry ID of that hit}. """
    return {series['track']: series['geometry_id']
            for _, series in truth_df.iterrows()}


def remove_low_pt_tracks(_df, _initial_df, min_pt_gev=0.5):
    """ Removes tracks with truth pt smaller than 0.5 GeV """
    low_pt_pids = []
    for row, series in _initial_df.iterrows():
        if series['pt'] < min_pt_gev:
            low_pt_pids.append(series['particle_id'])

    _df = _df[~_df['particle_id'].isin(low_pt_pids)]
    _initial_df = _initial_df[~_initial_df['particle_id'].isin(low_pt_pids)]

    return _df.copy().reset_index(), _initial_df.copy().reset_index()


def convert_tracks(hits, truth_df, xy_to_rz=True):
    """ Given a list of xy tracks, it returns a list with the corresponding
        rz tracks. """
    first_transform, second_transform = ('xy_track', 'rz_track') if xy_to_rz \
        else ('rz_track', 'xy_track')
    return [series[second_transform] for _, series in truth_df.iterrows()
            if series[first_transform] in hits]
