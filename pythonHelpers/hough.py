import ROOT
import numpy as np
import SndlhcMuonReco

def run_hough_transform(muon_reco_task, event, geo, z_vtx_min=None, z_vtx_max=None):
    """Identify tracks using Hough transform with vertex constraints."""
    hit_collection = {
        "pos": [[], [], []], "d": [[], [], []],
        "vert": [], "system": [], "detectorID": []
    }
    pos_a, pos_b = ROOT.TVector3(), ROOT.TVector3()

    for hit in event.Digi_ScifiHits:
        if not hit.isValid():
            continue

        geo.modules['Scifi'].GetSiPMPosition(hit.GetDetectorID(), pos_a, pos_b)
        for i in range(3):
            hit_collection["pos"][i].append(pos_a[i])

        hit_collection["d"][0].append(muon_reco_task.Scifi_dx)
        hit_collection["d"][1].append(muon_reco_task.Scifi_dy)
        hit_collection["d"][2].append(muon_reco_task.Scifi_dz)
        hit_collection["vert"].append(hit.isVertical())
        hit_collection["system"].append(0)
        hit_collection["detectorID"].append(hit.GetDetectorID())

    if not hit_collection['pos'][0]:
        return 0, {'XZ': [], 'YZ': []}

    for k in ["pos", "d"]:
        hit_collection[k] = np.array(hit_collection[k], dtype=np.float32)
    for k, dt in [("vert", np.bool_), ("system", np.int32), ("detectorID", np.int32)]:
        hit_collection[k] = np.array(hit_collection[k], dtype=dt)

    counts, lines = {'XZ': 0, 'YZ': 0}, {'XZ': [], 'YZ': []}

    for projection_name in ['XZ', 'YZ']:
        is_vertical, axis = (projection_name == 'XZ'), (0 if projection_name == 'XZ' else 1)
        hough_object = muon_reco_task.h_ZX if is_vertical else muon_reco_task.h_ZY
        hits_used = np.zeros(len(hit_collection['pos'][0]), dtype=np.bool_)

        valid_lines_found = 0
        attempts = 0
        max_attempts = 20

        while valid_lines_found < muon_reco_task.max_reco_muons and attempts < max_attempts:
            attempts += 1
            mask = np.logical_and(hit_collection['vert'] == is_vertical, ~hits_used)
            if not np.any(mask): break

            fit_result = hough_object.fit_randomize(
                np.dstack([hit_collection['pos'][2][mask], hit_collection['pos'][axis][mask]])[0],
                np.dstack([hit_collection['d'][2][mask], hit_collection['d'][axis][mask]])[0],
                muon_reco_task.n_random, False, False
            )

            if fit_result[0] in [-1, -999]:
                # print(f"DEBUG: fit_randomize returned {fit_result[0]} for {projection_name}")
                break

            new_slope, new_intercept = fit_result[0], fit_result[1]
            related_hits = SndlhcMuonReco.hit_finder(
                new_slope, new_intercept,
                np.dstack([hit_collection['pos'][2][mask], hit_collection['pos'][axis][mask]]),
                np.dstack([hit_collection['d'][2][mask], hit_collection['d'][axis][mask]]),
                muon_reco_task.tolerance
            )

            if len(related_hits) == 0: break

            n_planes = SndlhcMuonReco.numPlanesHit(hit_collection['system'][mask][related_hits], hit_collection['detectorID'][mask][related_hits])
            # print(f"DEBUG: {projection_name} track candidate: {n_planes} planes hit (min: {muon_reco_task.min_planes_hit})")
            if n_planes >= muon_reco_task.min_planes_hit:
                skip_track = False
                conflict_params = None

                for existing_line in lines[projection_name]:
                    ext_m, ext_c = existing_line[0], existing_line[1]
                    if abs(new_slope - ext_m) > 1e-6:
                        z_vertex = (ext_c - new_intercept) / (new_slope - ext_m)
                        if (z_vtx_min is not None and z_vertex < z_vtx_min) or (z_vtx_max is not None and z_vertex > z_vtx_max):
                            skip_track = True
                            conflict_params = (ext_m, ext_c)
                            break

                if skip_track:
                    if conflict_params:
                        global_indices = np.where(mask)[0][related_hits]
                        z_bad, c_bad = hit_collection['pos'][2][global_indices], hit_collection['pos'][axis][global_indices]
                        dist = np.abs(c_bad - (conflict_params[0] * z_bad + conflict_params[1]))
                        hits_used[global_indices[np.argmin(dist)]] = True
                    else:
                        hits_used[np.where(mask)[0][related_hits]] = True
                    continue

                counts[projection_name] += 1
                lines[projection_name].append(fit_result)

                selected_global_idx = np.where(mask)[0][related_hits]
                projection_idx = np.where(hit_collection['vert'] == is_vertical)[0]

                z_sel, c_sel = hit_collection['pos'][2][selected_global_idx], hit_collection['pos'][axis][selected_global_idx]
                z_all, c_all = hit_collection['pos'][2][projection_idx], hit_collection['pos'][axis][projection_idx]

                dz = z_all[:, np.newaxis] - z_sel[np.newaxis, :]
                dc = c_all[:, np.newaxis] - c_sel[np.newaxis, :]
                close_hits_mask = np.any((np.abs(dz) < 1e-3) & (np.abs(dc) < muon_reco_task.tolerance), axis=1)
                hits_used[projection_idx[close_hits_mask]] = True
                valid_lines_found += 1
            else:
                break

    return max(counts.values()), lines

def get_line_params(projection, index, track_lines):
    if index < len(track_lines[projection]):
        return track_lines[projection][index][0], track_lines[projection][index][1]
    return np.nan, np.nan
