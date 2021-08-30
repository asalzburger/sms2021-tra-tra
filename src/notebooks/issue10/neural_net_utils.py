import pandas as pd

from src.utils.metrics import get_track_to_truth_row_mapping, leading_particle


def compute_approximate_qpt_real_track(hits):
    """ Computes the truth track for the q/p_T transform, using the
        approximated formula: q/p_T = (1 / Ar) * phi_0 + phi_1 / Ar. """
    a = 3e-4
    (r1, phi_11), (r2, phi_12) = hits[0], hits[1]
    phi = (r2 * phi_11 - r1 * phi_12) / (r2 - r1)
    qpt = (phi - phi_11) / (a * r1)
    return phi, qpt


def compute_truth_tracks(truth_df, truth_compute_function):
    """ Computes the truth tracks for an event. """
    particles = set(truth_df['particle_id'])
    truth_tracks = {}
    for pid in particles:
        particle_df = truth_df[truth_df['particle_id'] == pid]
        tracks = list(particle_df['track'])
        x, y = truth_compute_function(tracks)
        truth_tracks[pid] = (x, y)
    return truth_tracks


def tracks_are_close(track1, track2, closeness_thresholds):
    """ Returns True if two tracks are within a range defined by the
        `closeness_thresholds` argument; Else False. """
    x_t = closeness_thresholds[0]
    y_t = closeness_thresholds[1]
    return abs(track1[0] - track2[0]) < x_t and abs(track1[1] - track2[1]) < y_t


def find_close_truth_tracks(truth_tracks, closeness_thresholds):
    """ For every truth track, a list of tracks that are close to it is returned
        in the form of a dictionary: {tracks: [list-of-tracks-close-to-it]} """
    closeness_mapping = {truth_track: [] for truth_track in truth_tracks}
    for idx, track1 in enumerate(truth_tracks):
        for track2 in truth_tracks[idx + 1:]:
            if tracks_are_close(track1, track2, closeness_thresholds):
                closeness_mapping[track1].append(track2)
    return closeness_mapping


def get_data_from_hough_transform(truth_df, est_tracks_to_hits):
    """ Returns a dataframe containing truth data regarding the results
        of the Hough Transform. """
    track_to_df_row = get_track_to_truth_row_mapping(truth_df)
    particles = set(truth_df['particle_id'])
    particle_to_tracks_finding_it = {pid: [] for pid in particles}
    for track, hits in est_tracks_to_hits.items():
        leading_pid = leading_particle(hits, track_to_df_row, truth_df)
        particle_to_tracks_finding_it[leading_pid].append(track)


"""
DataFrame could look like this:

>>> x = {'track1': [[0,1,2,3], [4,5,6]], 'track2': [[1,2,3,4], [5,6,7,8]], 
         'are_duplicates': [True, False], 'bin-size': [(0.04, 14), (0.001, 16)]}
>>> df = pd.DataFrame(x)
>>> df
         track1        track2  are_duplicates     bin-size
0  [0, 1, 2, 3]  [1, 2, 3, 4]            True   (0.04, 14)
1     [4, 5, 6]  [5, 6, 7, 8]           False  (0.001, 16)
"""
