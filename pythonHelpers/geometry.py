import ROOT
from array import array
import rootUtils as ut

def initialize_event_display(geo, histograms):
    """Setup the canvas and background histograms for event display."""
    ut.bookCanvas(histograms, key='simpleDisplay', title='simple event display', nx=1200, ny=1016, cx=1, cy=2)

    # Determine Z range based on detector floor position
    z_start = 250. if geo.snd_geo.Floor.z != 0 else 60.
    z_end = z_start + 350.
    x_start, y_start = -100., -30.

    histograms.update({
        'xmin': x_start, 'xmax': x_start + 110.,
        'ymin': y_start, 'ymax': y_start + 110.,
        'zmin': z_start, 'zmax': z_end
    })

    for projection, title in [('xz', '; z [cm]; x [cm]'), ('yz', '; z [cm]; y [cm]')]:
        ut.bookHist(histograms, projection, title, 500, histograms['zmin'], histograms['zmax'], 100, histograms[f'{projection[0]}min'], histograms[f'{projection[0]}max'])
        histograms[projection].SetStats(0)

def draw_detector_geometry(geo, histograms):
    """Draw geometry boxes for display."""
    mu_filter = geo.snd_geo.MuFilter
    scifi = geo.snd_geo.Scifi
    emulsion = geo.snd_geo.EmulsionDet
    navigator = ROOT.gGeoManager.GetCurrentNavigator()

    nodes = {'volMuFilter_1/volFeBlockEnd_1': ROOT.kGreen-6}

    for i in range(mu_filter.NVetoPlanes):
        nodes[f'volVeto_1/volVetoPlane_{i}_{i}'] = ROOT.kRed
        for j in range(mu_filter.NVetoBars):
            bar_suffix = 'Bar_1' if i < 2 else 'Bar_ver_1'
            nodes[f'volVeto_1/volVetoPlane_{i}_{i}/volVeto{bar_suffix}{i}{j:0>3d}'] = ROOT.kRed
        box_type = "3" if i == 2 else ""
        nodes[f'volVeto_1/subVeto{box_type}Box_{i}'] = ROOT.kGray+1

    for i in range(scifi.nscifi):
        nodes[f'volTarget_1/ScifiVolume{i+1}_{i+1}000000'] = ROOT.kBlue+1
        nodes[f'volTarget_1/volFeTarget{i+1}_1'] = ROOT.kGreen-6
    for i in range(emulsion.wall):
        nodes[f'volTarget_1/volWallborder_{i}'] = ROOT.kGray

    ds_offset = mu_filter.NVetoPlanes + mu_filter.NUpstreamPlanes
    for i in range(mu_filter.NDownstreamPlanes):
        ds_plane_name = f'volMuFilter_1/volMuDownstreamDet_{i}_{i+ds_offset}'
        nodes[ds_plane_name] = ROOT.kBlue+1
        for j in range(mu_filter.NDownstreamBars):
            nodes[f'{ds_plane_name}/volMuDownstreamBar_ver_3{i}{j+mu_filter.NDownstreamBars:0>3d}'] = ROOT.kBlue+1
            if i < 3:
                nodes[f'{ds_plane_name}/volMuDownstreamBar_hor_3{i}{j:0>3d}'] = ROOT.kBlue+1
        nodes[f'volMuFilter_1/subDSBox_{i+ds_offset}'] = ROOT.kGray+1

    for i in range(mu_filter.NUpstreamPlanes):
        nodes[f'volMuFilter_1/subUSBox_{i+mu_filter.NVetoPlanes}'] = ROOT.kGray+1
        nodes[f'volMuFilter_1/volMuUpstreamDet_{i}_{i+mu_filter.NVetoPlanes}'] = ROOT.kBlue+1
        for j in range(mu_filter.NUpstreamBars):
            nodes[f'volMuFilter_1/volMuUpstreamDet_{i}_{i+mu_filter.NVetoPlanes}/volMuUpstreamBar_2{i}00{j}'] = ROOT.kBlue+1
        nodes[f'volMuFilter_1/volFeBlock_{i}'] = ROOT.kGreen-6

    for i in range(ds_offset, ds_offset + mu_filter.NDownstreamPlanes):
        nodes[f'volMuFilter_1/volFeBlock_{i}'] = ROOT.kGreen-6

    pass_nodes = {'Block', 'Wall', 'FeTarget'}
    transverse_nodes = {'UpstreamBar', 'VetoBar', 'hor'}
    axis_map = {'X': 0, 'Y': 1}

    for node_path, color in nodes.items():
        full_path = f'/cave_1/Detector_0/{node_path}'
        for view in ['X', 'Y']:
            if not navigator.CheckPath(full_path):
                continue

            navigator.cd(full_path)
            shape = navigator.GetCurrentNode().GetVolume().GetShape()
            dx, dy, dz = shape.GetDX(), shape.GetDY(), shape.GetDZ()
            ox, oy, oz = shape.GetOrigin()[0], shape.GetOrigin()[1], shape.GetOrigin()[2]

            corners = {}
            if view == 'X' and (not any(tn in full_path for tn in transverse_nodes) or 'VetoBar_ver' in full_path):
                corners = {'LB': [-dx+ox, oy, -dz+oz], 'LT': [dx+ox, oy, -dz+oz], 'RB': [-dx+ox, oy, dz+oz], 'RT': [dx+ox, oy, dz+oz]}
            elif view == 'Y' and 'ver' not in full_path:
                corners = {'LB': [ox, -dy+oy, -dz+oz], 'LT': [ox, dy+oy, -dz+oz], 'RB': [ox, -dy+oy, dz+oz], 'RT': [ox, dy+oy, dz+oz]}

            if not corners: continue

            polyline = ROOT.TPolyLine()
            projection_axis = axis_map[view]
            for i, corner_key in enumerate(['LB', 'LT', 'RT', 'RB', 'LB']):
                master_coords = array('d', [0, 0, 0])
                navigator.LocalToMaster(array('d', corners[corner_key]), master_coords)
                polyline.SetPoint(i, master_coords[2], master_coords[projection_axis])

            polyline.SetLineColor(color)
            polyline.SetLineWidth(1)
            histograms['simpleDisplay'].cd(projection_axis + 1)

            if any(pn in full_path for pn in pass_nodes):
                polyline.SetFillColorAlpha(color, 0.5)
                polyline.Draw('f&&same')
            polyline.Draw('same')
            histograms[f"{full_path}_{view}"] = polyline

def draw_event_hits_and_tracks(event, geo, hough_lines, histograms, projections):
    """Draw hits and hough lines on the canvas."""
    for p_idx in projections:
        histograms['simpleDisplay'].cd(p_idx)
        histograms[projections[p_idx]].Draw('b')

    draw_detector_geometry(geo, histograms)

    hit_graphs = {view: {system: ROOT.TGraphErrors() for system in ['Scifi', 'MuFilter']} for view in ['X', 'Y']}
    counts = {view: {system: 0 for system in ['Scifi', 'MuFilter']} for view in ['X', 'Y']}

    scifi_geo = geo.snd_geo.Scifi
    for hit in event.Digi_ScifiHits:
        if not hit.isValid(): continue
        pos_a, pos_b = ROOT.TVector3(), ROOT.TVector3()
        geo.modules['Scifi'].GetSiPMPosition(hit.GetDetectorID(), pos_a, pos_b)
        view = 'X' if hit.isVertical() else 'Y'
        graph, idx = hit_graphs[view]['Scifi'], counts[view]['Scifi']
        graph.SetPoint(idx, pos_a.Z(), pos_a.X() if view == 'X' else pos_a.Y())
        graph.SetPointError(idx, scifi_geo.scifimat_z/2, scifi_geo.channel_width/2)
        counts[view]['Scifi'] += 1

    mu_filter_geo = geo.snd_geo.MuFilter
    for hit in event.Digi_MuFilterHits:
        if not hit.isValid(): continue
        pos_a, pos_b = ROOT.TVector3(), ROOT.TVector3()
        geo.modules['MuFilter'].GetPosition(hit.GetDetectorID(), pos_a, pos_b)
        view = 'X' if hit.isVertical() else 'Y'
        graph, idx = hit_graphs[view]['MuFilter'], counts[view]['MuFilter']
        system = hit.GetSystem()

        if system == 1:
            dx, dy, dz = (mu_filter_geo.Veto3BarX/2 if hasattr(mu_filter_geo, "Veto3BarX") else mu_filter_geo.VetoBarX/2), mu_filter_geo.VetoBarY/2, mu_filter_geo.VetoBarZ/2
        elif system == 2:
            dx, dy, dz = mu_filter_geo.UpstreamBarX/2, mu_filter_geo.UpstreamBarY/2, mu_filter_geo.UpstreamBarZ/2
        else: # DS
            dx, dy, dz = mu_filter_geo.DownstreamBarX_ver/2, mu_filter_geo.DownstreamBarY/2, mu_filter_geo.DownstreamBarZ/2

        graph.SetPoint(idx, (pos_a.Z() + pos_b.Z())/2, pos_a.X() if view == 'X' else pos_a.Y())
        graph.SetPointError(idx, dz, dx if view == 'X' else dy)
        counts[view]['MuFilter'] += 1

    for view, p_idx in [('X', 1), ('Y', 2)]:
        histograms['simpleDisplay'].cd(p_idx)
        for system, color, style in [('Scifi', ROOT.kBlue+2, 20), ('MuFilter', ROOT.kRed+1, 21)]:
            g = hit_graphs[view][system]
            g.SetMarkerStyle(style)
            g.SetMarkerSize(0.8)
            g.SetMarkerColor(color)
            g.Draw('sameP')
            histograms[f'hits_{system.lower()[:2]}_{view}'] = g

        proj_name = f'{view}Z'
        for i, (slope, intercept) in enumerate(hough_lines[proj_name]):
            z_min, z_max = histograms['zmin'], histograms['zmax']
            line = ROOT.TLine(z_min, slope * z_min + intercept, z_max, slope * z_max + intercept)
            line.SetLineColor(ROOT.kCyan+2); line.SetLineWidth(2)
            line.Draw("same")
            histograms[f'line_{proj_name}_{i}'] = line
