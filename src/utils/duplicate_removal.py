from tqdm.notebook import tqdm


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
    for idx, t1 in tqdm(enumerate(tracks), total=len(tracks),
                        desc='Duplicate Removal 1'):
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
    for idx, t1 in tqdm(enumerate(tracks), total=len(tracks),
                        desc='Duplicate Removal 2'):
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
