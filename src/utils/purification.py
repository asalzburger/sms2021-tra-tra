from tqdm.notebook import tqdm

from src.utils.hough_transform import compute_b, hough2d_pipeline
from src.utils.preprocessing import convert_tracks


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


def purify_xy_estimations(xy_est_tracks_to_hits, rz_hyperparams, truth_df):
    """ Purifies the xy-estimated hits-per-bins by running the rz
        Hough Transform on the hits inside every bin (discretely). """
    purified_estimations = {}
    for est_track, hits in tqdm(xy_est_tracks_to_hits.items(),
                                total=len(xy_est_tracks_to_hits),
                                desc='Purification of xy estimated tracks'):
        purified_hits = purify_xy_hits(hits, rz_hyperparams, truth_df)
        if len(purified_hits) > 0:
            purified_estimations[est_track] = purified_hits
    return purified_estimations
