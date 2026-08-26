#!/usr/bin/env python3
"""
filter_and_display_tridents.py
------------------------------
Generates 2D (XZ, YZ) and 3D event displays for all events in input ROOT files
without filtering, adopting the official SND@LHC 2dEventDisplay.py detector & hit
geometry methodology, and saving the displays directly into output ROOT files.
"""

import os
import sys
import glob
import time
import array
import argparse
import ROOT
import SndlhcGeo

ROOT.gROOT.SetBatch(True)

# Geometry and coordinate constants
Z_3D_MIN = 0.0
Z_MAX = 600.0
ROCK_BOUNDARY = 360.0

def make_array(lst):
    return array.array('d', lst)

def style_detector_volumes(vol):
    """Recursively styles TGeo volumes for standard 3D representation."""
    for i in range(vol.GetNdaughters()):
        dnode = vol.GetNode(i)
        dvol = dnode.GetVolume()
        ROOT.SetOwnership(dvol, False)
        name = dnode.GetName()
        vname = dvol.GetName()

        dvol.SetVisibility(True)
        if 'Wall' in name and 'border' not in name:
            dvol.SetLineColor(ROOT.kGray+2)
            dvol.SetFillColor(ROOT.kGray+1)
            dvol.SetTransparency(30)
        elif 'Scifi' in name or 'ScifiVolume' in vname:
            dvol.SetLineColor(ROOT.kCyan-2)
            dvol.SetFillColor(ROOT.kCyan-6)
            dvol.SetTransparency(20)
        elif 'volFeBlock' in name or 'volFeTarget' in name:
            dvol.SetLineColor(ROOT.kGreen-3)
            dvol.SetFillColor(ROOT.kGreen-7)
            dvol.SetTransparency(20)
        elif 'volMuUpstreamDet' in name or 'volMuDownstreamDet' in name:
            dvol.SetLineColor(ROOT.kBlue-3)
            dvol.SetFillColor(ROOT.kBlue-7)
            dvol.SetTransparency(20)
        elif 'Veto' in name:
            dvol.SetLineColor(ROOT.kOrange+2)
            dvol.SetFillColor(ROOT.kOrange-3)
            dvol.SetTransparency(20)
        else:
            dvol.SetLineColor(ROOT.kGray)
            dvol.SetTransparency(60)

        if dvol.GetNdaughters() > 0:
            style_detector_volumes(dvol)

def find_geofile(geofile_filename):
    """Searches multiple candidate locations for geofile to ensure it is always found."""
    if not geofile_filename:
        geofile_filename = "geofile_full.Ntuple-TGeant4_boost100.0.root"

    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    sndsw_root = os.environ.get("SNDSW_ROOT", "")

    candidates = [
        geofile_filename,
        os.path.basename(geofile_filename),
        os.path.join(script_dir, geofile_filename),
        os.path.join(script_dir, os.path.basename(geofile_filename)),
        os.path.join(sndsw_root, geofile_filename),
        os.path.join(sndsw_root, "python", os.path.basename(geofile_filename)),
        os.path.join(sndsw_root, "geofile_full.Ntuple-TGeant4_boost100.0.root"),
        "/afs/cern.ch/user/i/idioniso/snd_master/sndsw/python/geofile_full.Ntuple-TGeant4_boost100.0.root",
        "/eos/experiment/sndlhc/convertedData/physics/2022/geofile_sndlhc_TI18_V0_2022.root"
    ]

    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)

    return geofile_filename

def load_detector_geometry(geofile_path):
    """
    Loads detector geometry using SndlhcGeo.GeoInterface.
    Extracts 2D bounding boxes and 3D visual structures.
    """
    geofile_path = find_geofile(geofile_path)
    if not os.path.exists(geofile_path):
        print(f"Warning: Geofile not found at '{geofile_path}'.")
        return [], None, None, None, None

    snd_geo = SndlhcGeo.GeoInterface(geofile_path)
    f_geo = ROOT.TFile.Open(geofile_path, "READ")
    if not f_geo or f_geo.IsZombie():
        print(f"Warning: Could not open geometry file '{geofile_path}'.")
        return [], None, None, None, None

    ggeo = ROOT.gGeoManager
    top_vol = ggeo.GetTopVolume()
    ROOT.SetOwnership(top_vol, False)
    style_detector_volumes(top_vol)

    scifi_module = snd_geo.modules.get('Scifi', None)
    mufilter_module = snd_geo.modules.get('MuFilter', None)
    nav = ggeo.GetCurrentNavigator()

    geo_elements = []

    def extract_node_bounds(node_path, label, color, line_color=None, is_filled=True):
        if not nav.CheckPath(node_path):
            return
        nav.cd(node_path)
        node = nav.GetCurrentNode()
        vol = node.GetVolume()
        shape = vol.GetShape()
        dx, dy, dz = shape.GetDX(), shape.GetDY(), shape.GetDZ()
        ox, oy, oz = shape.GetOrigin()[0], shape.GetOrigin()[1], shape.GetOrigin()[2]

        corners = [
            array.array('d', [-dx + ox, -dy + oy, -dz + oz]),
            array.array('d', [ dx + ox, -dy + oy, -dz + oz]),
            array.array('d', [-dx + ox,  dy + oy, -dz + oz]),
            array.array('d', [ dx + ox,  dy + oy, -dz + oz]),
            array.array('d', [-dx + ox, -dy + oy,  dz + oz]),
            array.array('d', [ dx + ox, -dy + oy,  dz + oz]),
            array.array('d', [-dx + ox,  dy + oy,  dz + oz]),
            array.array('d', [ dx + ox,  dy + oy,  dz + oz]),
        ]

        master_pts = []
        for c in corners:
            m = array.array('d', [0.0, 0.0, 0.0])
            nav.LocalToMaster(c, m)
            master_pts.append(m)

        xs = [p[0] for p in master_pts]
        ys = [p[1] for p in master_pts]
        zs = [p[2] for p in master_pts]

        geo_elements.append({
            'name': node_path,
            'label': label,
            'color': color,
            'line_color': line_color if line_color is not None else color,
            'is_filled': is_filled,
            'x': (min(xs), max(xs)),
            'y': (min(ys), max(ys)),
            'z': (min(zs), max(zs)),
        })

    # Veto Detector (clean orange outline)
    for i in range(2):
        extract_node_bounds(f"/cave_1/Detector_0/volVeto_1/subVetoBox_{i}", "Veto Detector", ROOT.kOrange+2, ROOT.kOrange+2, is_filled=False)

    # SciFi Target Emulsion Walls (filled light grey)
    for i in range(5):
        extract_node_bounds(f"/cave_1/Detector_0/volTarget_1/volWallborder_{i}", "Target Emulsion Wall", ROOT.kGray, ROOT.kGray+2, is_filled=True)

    # SciFi Stations (clean blue outline)
    for i in range(1, 6):
        extract_node_bounds(f"/cave_1/Detector_0/volTarget_1/ScifiVolume{i}_{i}000000", "SciFi Station", ROOT.kAzure-4, ROOT.kAzure-4, is_filled=False)

    # MuFilter Upstream (5 Iron Blocks filled + 5 Active Planes outline)
    for i in range(5):
        extract_node_bounds(f"/cave_1/Detector_0/volMuFilter_1/volFeBlock_{i}", "MuFilter Iron Block", ROOT.kGreen-9, ROOT.kGreen-3, is_filled=True)
        extract_node_bounds(f"/cave_1/Detector_0/volMuFilter_1/volMuUpstreamDet_{i}_{i+2}", "MuFilter Active Plane", ROOT.kBlue-4, ROOT.kBlue-4, is_filled=False)

    # MuFilter Downstream (3 Iron Blocks filled: 7, 8, 9 + 4 Active Planes outline: 0..3)
    for fe_id in [7, 8, 9]:
        extract_node_bounds(f"/cave_1/Detector_0/volMuFilter_1/volFeBlock_{fe_id}", "MuFilter Iron Block", ROOT.kGreen-9, ROOT.kGreen-3, is_filled=True)

    for i in range(4):
        extract_node_bounds(f"/cave_1/Detector_0/volMuFilter_1/volMuDownstreamDet_{i}_{i+7}", "MuFilter Active Plane", ROOT.kBlue-4, ROOT.kBlue-4, is_filled=False)

    # MuFilter End Iron Block (filled)
    extract_node_bounds("/cave_1/Detector_0/volMuFilter_1/volFeBlockEnd_1", "MuFilter Iron Block", ROOT.kGreen-9, ROOT.kGreen-3, is_filled=True)

    print(f"Loaded {len(geo_elements)} detector geometry elements from '{geofile_path}'.")
    return geo_elements, ggeo, f_geo, scifi_module, mufilter_module, snd_geo

def build_mc_points_map(event):
    """
    Collects true simulated detector steps from ScifiPoint, MuFilterPoint, and EmulsionDetPoint.
    Maps trackID -> list of (z, x, y, pz, px, py) sorted along Z.
    """
    mc_points_map = {}
    for col_name in ['ScifiPoint', 'MuFilterPoint', 'EmulsionDetPoint']:
        if hasattr(event, col_name):
            col = getattr(event, col_name)
            for pt in col:
                trID = pt.GetTrackID()
                if trID not in mc_points_map:
                    mc_points_map[trID] = []
                mc_points_map[trID].append((pt.GetZ(), pt.GetX(), pt.GetY(), pt.GetPz(), pt.GetPx(), pt.GetPy()))
    for trID in mc_points_map:
        mc_points_map[trID].sort(key=lambda p: p[0])
    return mc_points_map

def get_track_trajectory_points(tr_i, track, z_min_plot, z_max_plot, mc_points_map=None):
    """
    Calculates polyline coordinates (x_arr, y_arr, z_arr) for a track.
    Uses true simulated Geant4 step coordinates in the detector to account for multiple
    Coulomb scattering through the rock. For non-muon shower products, tracks are terminated
    at their true absorption point rather than artificially extrapolated through the detector.
    """
    z0, x0, y0 = track.GetStartZ(), track.GetStartX(), track.GetStartY()
    pz, px, py = track.GetPz(), track.GetPx(), track.GetPy()
    pdg = track.GetPdgCode()
    is_muon = (abs(pdg) == 13)

    det_pts = mc_points_map.get(tr_i, []) if (mc_points_map is not None and tr_i is not None and tr_i >= 0) else []
    pts = []

    if det_pts:
        first_pt = det_pts[0]
        if z0 >= z_min_plot:
            pts.append((z0, x0, y0))
        else:
            pz_f, px_f, py_f = first_pt[3], first_pt[4], first_pt[5]
            if pz_f != 0:
                x_enter = first_pt[1] + (px_f / pz_f) * (z_min_plot - first_pt[0])
                y_enter = first_pt[2] + (py_f / pz_f) * (z_min_plot - first_pt[0])
            else:
                x_enter, y_enter = first_pt[1], first_pt[2]
            pts.append((z_min_plot, x_enter, y_enter))

        for p in det_pts:
            if not pts or abs(p[0] - pts[-1][0]) > 0.05:
                pts.append((p[0], p[1], p[2]))

        last_pt = det_pts[-1]
        pz_l, px_l, py_l = last_pt[3], last_pt[4], last_pt[5]
        # Only penetrating muons that reach the end of the active volume are extrapolated forward
        if is_muon and last_pt[0] < z_max_plot and pz_l > 0:
            x_exit = last_pt[1] + (px_l / pz_l) * (z_max_plot - last_pt[0])
            y_exit = last_pt[2] + (py_l / pz_l) * (z_max_plot - last_pt[0])
            pts.append((z_max_plot, x_exit, y_exit))
    else:
        if is_muon and pz != 0:
            x_start = x0 if z0 >= z_min_plot else x0 + (px / pz) * (z_min_plot - z0)
            y_start = y0 if z0 >= z_min_plot else y0 + (py / pz) * (z_min_plot - z0)
            z_s = max(z0, z_min_plot)
            x_end = x0 + (px / pz) * (z_max_plot - z0)
            y_end = y0 + (py / pz) * (z_max_plot - z0)
            pts.append((z_s, x_start, y_start))
            pts.append((z_max_plot, x_end, y_end))
        elif z0 >= z_min_plot and pz > 0:
            # Short local trajectory for non-penetrating shower particle produced in detector
            p_tot = (px**2 + py**2 + pz**2)**0.5
            dr = min(20.0, max(2.0, p_tot * 50.0))
            pts.append((z0, x0, y0))
            pts.append((z0 + (pz / p_tot) * dr, x0 + (px / p_tot) * dr, y0 + (py / p_tot) * dr))

    z_arr = [p[0] for p in pts]
    x_arr = [p[1] for p in pts]
    y_arr = [p[2] for p in pts]
    return x_arr, y_arr, z_arr

def extract_file_tag(filename):
    """Extracts tag such as digCPP-200 from filename."""
    base = os.path.basename(filename)
    if "digCPP-" in base:
        parts = base.split("digCPP-")
        if len(parts) > 1:
            tag = "digCPP-" + parts[1].replace(".root", "")
            return tag
    return "mc"

def make_box_poly(zmin, zmax, vmin, vmax):
    """Creates a 5-point closed TPolyLine rectangle."""
    zs = array.array('d', [zmin, zmax, zmax, zmin, zmin])
    vs = array.array('d', [vmin, vmin, vmax, vmax, vmin])
    return ROOT.TPolyLine(5, zs, vs)

def draw_detector_geometry(geo_elements, z_range, x_range, y_range, ggeo=None, file_tag="", evt_num=0, i_event=0):
    """
    Initializes 2D (XZ, YZ) and 3D canvases and draws detector geometry boundaries.
    Uses TPolyLine for 2D boxes to guarantee correct painter depth stacking in both TBrowser and VSCode/JSROOT.
    """
    z_min_plot, z_max_plot = z_range
    x_min_plot, x_max_plot = x_range
    y_min_plot, y_max_plot = y_range

    # ----------------- XZ Projection -----------------
    c_xz = ROOT.TCanvas(f"XZ_Ev_{evt_num}_F_{file_tag}_Idx_{i_event}", f"XZ Projection (Event #{evt_num})", 950, 650)
    ROOT.SetOwnership(c_xz, False)
    frame_xz = c_xz.DrawFrame(z_min_plot, x_min_plot, z_max_plot, x_max_plot, ";Z [cm];X [cm]")
    ROOT.SetOwnership(frame_xz, False)
    c_xz.Update()

    boxes_xz = []
    legend_xz = ROOT.TLegend(0.12, 0.70, 0.44, 0.89)
    ROOT.SetOwnership(legend_xz, False)
    legend_xz.SetBorderSize(1)
    legend_xz.SetFillStyle(1001)
    legend_xz.SetFillColorAlpha(ROOT.kWhite, 0.88)
    legend_xz.SetTextFont(42)
    legend_xz.SetTextSize(0.024)

    for elem in geo_elements:
        poly = make_box_poly(elem['z'][0], elem['z'][1], elem['x'][0], elem['x'][1])
        ROOT.SetOwnership(poly, False)
        poly.SetLineColor(elem.get('line_color', elem['color']))
        poly.SetLineWidth(1)
        if elem.get('is_filled', True):
            poly.SetFillStyle(1001)
            poly.SetFillColorAlpha(elem['color'], 0.4)
            poly.Draw("f")
        poly.Draw()
        boxes_xz.append(poly)

    # ----------------- YZ Projection -----------------
    c_yz = ROOT.TCanvas(f"YZ_Ev_{evt_num}_F_{file_tag}_Idx_{i_event}", f"YZ Projection (Event #{evt_num})", 950, 650)
    ROOT.SetOwnership(c_yz, False)
    frame_yz = c_yz.DrawFrame(z_min_plot, y_min_plot, z_max_plot, y_max_plot, ";Z [cm];Y [cm]")
    ROOT.SetOwnership(frame_yz, False)
    c_yz.Update()

    boxes_yz = []
    legend_yz = ROOT.TLegend(0.12, 0.70, 0.44, 0.89)
    ROOT.SetOwnership(legend_yz, False)
    legend_yz.SetBorderSize(1)
    legend_yz.SetFillStyle(1001)
    legend_yz.SetFillColorAlpha(ROOT.kWhite, 0.88)
    legend_yz.SetTextFont(42)
    legend_yz.SetTextSize(0.024)

    for elem in geo_elements:
        poly = make_box_poly(elem['z'][0], elem['z'][1], elem['y'][0], elem['y'][1])
        ROOT.SetOwnership(poly, False)
        poly.SetLineColor(elem.get('line_color', elem['color']))
        poly.SetLineWidth(1)
        if elem.get('is_filled', True):
            poly.SetFillStyle(1001)
            poly.SetFillColorAlpha(elem['color'], 0.4)
            poly.Draw("f")
        poly.Draw("")
        boxes_yz.append(poly)

    # ----------------- 3D View -----------------
    c_3d = ROOT.TCanvas(f"3D_Ev_{evt_num}_F_{file_tag}_Idx_{i_event}", f"3D View (Event #{evt_num})", 850, 850)
    ROOT.SetOwnership(c_3d, False)

    h3 = ROOT.TH3F(f"axis_box_3d_{evt_num}_F_{file_tag}_Idx_{i_event}", ";Z [cm];X [cm];Y [cm]",
                   10, Z_3D_MIN, Z_MAX,
                   10, x_min_plot, x_max_plot,
                   10, y_min_plot, y_max_plot)
    ROOT.SetOwnership(h3, False)
    h3.SetStats(0)
    h3.Draw()

    geo_3d_lines = []
    for elem in geo_elements:
        xmin, xmax = elem['x']
        ymin, ymax = elem['y']
        zmin, zmax = elem['z']
        corners = [
            (zmin, xmin, ymin), (zmax, xmin, ymin), (zmax, xmax, ymin), (zmin, xmax, ymin), (zmin, xmin, ymin),
            (zmin, xmin, ymax), (zmax, xmin, ymax), (zmax, xmax, ymax), (zmin, xmax, ymax), (zmin, xmin, ymax)
        ]
        pl = ROOT.TPolyLine3D(10)
        ROOT.SetOwnership(pl, False)
        for idx, (z, x, y) in enumerate(corners):
            pl.SetPoint(idx, z, x, y)
        pl.SetLineColor(elem.get('line_color', elem['color']))
        pl.SetLineWidth(1)
        pl.Draw()
        geo_3d_lines.append(pl)

        verticals = [
            ((zmax, xmin, ymin), (zmax, xmin, ymax)),
            ((zmax, xmax, ymin), (zmax, xmax, ymax)),
            ((zmin, xmax, ymin), (zmin, xmax, ymax))
        ]
        for (p1, p2) in verticals:
            pl_edge = ROOT.TPolyLine3D(2)
            ROOT.SetOwnership(pl_edge, False)
            pl_edge.SetPoint(0, p1[0], p1[1], p1[2])
            pl_edge.SetPoint(1, p2[0], p2[1], p2[2])
            pl_edge.SetLineColor(elem.get('line_color', elem['color']))
            pl_edge.SetLineWidth(1)
            pl_edge.Draw()
            geo_3d_lines.append(pl_edge)

    return {
        'c_xz': c_xz, 'c_yz': c_yz, 'c_3d': c_3d,
        'boxes_xz': boxes_xz, 'boxes_yz': boxes_yz,
        'legend_xz': legend_xz, 'legend_yz': legend_yz,
        'geo_3d_lines': geo_3d_lines,
        'h3': h3
    }

def draw_detector_hits(event, scifi_module, mufilter_module, geo_ctx, snd_geo=None):
    """
    Reconstructs and plots digitized detector hits from Digi_ScifiHits and Digi_MuFilterHits.
    Adopts official 2dEventDisplay.py conventions:
    - SciFi hits: Cluster centroids plotted as blue markers (ROOT.kBlue+2) with channel error bars.
    - MuFilter / Veto hits: Drawn as solid black TPolyLine boxes (ROOT.kBlack).
      * Horizontal bars (measuring Y, such as Upstream and horizontal Veto): drawn ONLY on YZ projection.
      * Vertical bars (measuring X, such as Downstream vertical): drawn ONLY on XZ projection.
    """
    c_xz = geo_ctx['c_xz']
    c_yz = geo_ctx['c_yz']
    c_3d = geo_ctx['c_3d']
    legend_xz = geo_ctx['legend_xz']
    legend_yz = geo_ctx['legend_yz']

    A = ROOT.TVector3()
    B = ROOT.TVector3()

    si = snd_geo.snd_geo.Scifi if (snd_geo and hasattr(snd_geo, "snd_geo") and hasattr(snd_geo.snd_geo, "Scifi")) else None
    sY_scifi = si.channel_width if si else 0.025
    sZ_scifi = si.scifimat_z if si else 0.135

    mi = snd_geo.snd_geo.MuFilter if (snd_geo and hasattr(snd_geo, "snd_geo") and hasattr(snd_geo.snd_geo, "MuFilter")) else None
    vetoYdim = (mi.VetoBarY / 2.0) if (mi and hasattr(mi, "VetoBarY")) else 3.0
    vetoZdim = (mi.VetoBarZ / 2.0) if (mi and hasattr(mi, "VetoBarZ")) else 1.0
    usYdim = (mi.UpstreamBarY / 2.0) if (mi and hasattr(mi, "UpstreamBarY")) else 4.0
    usZdim = (mi.UpstreamBarZ / 2.0) if (mi and hasattr(mi, "UpstreamBarZ")) else 1.5
    dsXdim = (mi.DownstreamBarX_ver / 2.0) if (mi and hasattr(mi, "DownstreamBarX_ver")) else 1.0
    dsYdim = (mi.DownstreamBarY / 2.0) if (mi and hasattr(mi, "DownstreamBarY")) else 4.0
    dsZdim = (mi.DownstreamBarZ / 2.0) if (mi and hasattr(mi, "DownstreamBarZ")) else 1.5

    gr_scifi_xz = ROOT.TGraphErrors()
    gr_scifi_yz = ROOT.TGraphErrors()
    ROOT.SetOwnership(gr_scifi_xz, False)
    ROOT.SetOwnership(gr_scifi_yz, False)

    hit_boxes_xz = []
    hit_boxes_yz = []
    poly3d_list = []

    # --- 1. SciFi Hits ---
    n_scifi_x = 0
    n_scifi_y = 0

    if scifi_module and hasattr(event, "Digi_ScifiHits"):
        for hit in event.Digi_ScifiHits:
            detID = hit.GetDetectorID()
            scifi_module.GetSiPMPosition(detID, A, B)

            if hit.isVertical():
                gr_scifi_xz.SetPoint(n_scifi_x, A.Z(), A.X())
                gr_scifi_xz.SetPointError(n_scifi_x, sZ_scifi, sY_scifi)
                n_scifi_x += 1
            else:
                gr_scifi_yz.SetPoint(n_scifi_y, A.Z(), A.Y())
                gr_scifi_yz.SetPointError(n_scifi_y, sZ_scifi, sY_scifi)
                n_scifi_y += 1

            pl3d = ROOT.TPolyLine3D(2)
            ROOT.SetOwnership(pl3d, False)
            pl3d.SetPoint(0, A.Z(), A.X(), A.Y())
            pl3d.SetPoint(1, B.Z(), B.X(), B.Y())
            pl3d.SetLineColorAlpha(ROOT.kAzure-4, 0.35)
            pl3d.SetLineWidth(1)
            poly3d_list.append(pl3d)

    # --- 2. Veto & MuFilter Hits (Orientation-Aware Solid Black Boxes) ---
    n_mufi_x = 0
    n_mufi_y = 0
    if mufilter_module and hasattr(event, "Digi_MuFilterHits"):
        for hit in event.Digi_MuFilterHits:
            detID = hit.GetDetectorID()
            system = hit.GetSystem()
            is_vert = hit.isVertical()
            mufilter_module.GetPosition(detID, A, B)

            z_mid = (A.Z() + B.Z()) / 2.0
            x_mid = (A.X() + B.X()) / 2.0
            y_mid = (A.Y() + B.Y()) / 2.0

            sZ = vetoZdim if system == 1 else (usZdim if system == 2 else dsZdim)

            if is_vert:
                # Vertical bar (measuring X) -> draw ONLY on XZ projection
                sX = dsXdim
                c_xz.cd()
                bx = make_box_poly(z_mid - sZ, z_mid + sZ, x_mid - sX, x_mid + sX)
                ROOT.SetOwnership(bx, False)
                bx.SetFillStyle(1001)
                bx.SetFillColor(ROOT.kBlack)
                bx.SetLineColor(ROOT.kBlack)
                bx.SetLineWidth(1)
                bx.Draw("f same")
                bx.Draw("same")
                hit_boxes_xz.append(bx)
                n_mufi_x += 1
            else:
                # Horizontal bar (measuring Y, like Upstream & Veto) -> draw ONLY on YZ projection
                sY = vetoYdim if system == 1 else (usYdim if system == 2 else dsYdim)
                c_yz.cd()
                by = make_box_poly(z_mid - sZ, z_mid + sZ, y_mid - sY, y_mid + sY)
                ROOT.SetOwnership(by, False)
                by.SetFillStyle(1001)
                by.SetFillColor(ROOT.kBlack)
                by.SetLineColor(ROOT.kBlack)
                by.SetLineWidth(1)
                by.Draw("f same")
                by.Draw("same")
                hit_boxes_yz.append(by)
                n_mufi_y += 1

            pl3d = ROOT.TPolyLine3D(2)
            ROOT.SetOwnership(pl3d, False)
            pl3d.SetPoint(0, A.Z(), A.X(), A.Y())
            pl3d.SetPoint(1, B.Z(), B.X(), B.Y())
            pl3d.SetLineColorAlpha(ROOT.kGray+2, 0.35)
            pl3d.SetLineWidth(1)
            poly3d_list.append(pl3d)

    # --- Draw onto XZ Canvas ---
    c_xz.cd()
    if n_scifi_x > 0:
        gr_scifi_xz.SetMarkerStyle(20)
        gr_scifi_xz.SetMarkerSize(1.2)
        gr_scifi_xz.SetMarkerColor(ROOT.kBlue+2)
        gr_scifi_xz.SetLineColor(ROOT.kBlue+2)
        gr_scifi_xz.Draw("sameP")
        legend_xz.AddEntry(gr_scifi_xz, "SciFi Hits", "p")

    if n_mufi_x > 0:
        leg_box_xz = ROOT.TBox()
        ROOT.SetOwnership(leg_box_xz, False)
        leg_box_xz.SetFillStyle(1001)
        leg_box_xz.SetFillColor(ROOT.kBlack)
        leg_box_xz.SetLineColor(ROOT.kBlack)
        legend_xz.AddEntry(leg_box_xz, "MuFilter / Veto Hits", "f")

    # --- Draw onto YZ Canvas ---
    c_yz.cd()
    if n_scifi_y > 0:
        gr_scifi_yz.SetMarkerStyle(20)
        gr_scifi_yz.SetMarkerSize(1.2)
        gr_scifi_yz.SetMarkerColor(ROOT.kBlue+2)
        gr_scifi_yz.SetLineColor(ROOT.kBlue+2)
        gr_scifi_yz.Draw("sameP")
        legend_yz.AddEntry(gr_scifi_yz, "SciFi Hits", "p")

    if n_mufi_y > 0:
        leg_box_yz = ROOT.TBox()
        ROOT.SetOwnership(leg_box_yz, False)
        leg_box_yz.SetFillStyle(1001)
        leg_box_yz.SetFillColor(ROOT.kBlack)
        leg_box_yz.SetLineColor(ROOT.kBlack)
        legend_yz.AddEntry(leg_box_yz, "MuFilter / Veto Hits", "f")

    # --- Draw onto 3D Canvas ---
    c_3d.cd()
    for pl in poly3d_list:
        pl.Draw("same")

    return {
        'gr_scifi_xz': gr_scifi_xz, 'gr_scifi_yz': gr_scifi_yz,
        'hit_boxes_xz': hit_boxes_xz, 'hit_boxes_yz': hit_boxes_yz,
        'poly3d_list': poly3d_list
    }

def identify_trident_topology(event):
    """
    Reconstructs the full physical path from the primary particle down to the trident muon pair,
    including all intermediate in-between MCTracks (radiating muons, brem photons, EM shower e+, hadrons).
    Returns a dictionary describing the complete genealogy and process type.
    """
    if not hasattr(event, "MCTrack") or not event.MCTrack:
        return None

    tracks = list(event.MCTrack)

    # 1. Primary track (MotherId == -1)
    primary_idx = -1
    primary_track = None
    for i, tr in enumerate(tracks):
        if tr.GetMotherId() == -1:
            primary_idx = i
            primary_track = tr
            break

    # 2. Find candidate daughter pairs (mu+ and mu-)
    pair_daughters = []
    for i, tr in enumerate(tracks):
        pdg = tr.GetPdgCode()
        proc = tr.GetProcID()
        if abs(pdg) == 13 and (proc in [5, 11] or tr.GetMotherId() != -1):
            pair_daughters.append((i, tr.GetMotherId(), pdg, tr))

    mothers = {}
    for i, m_id, pdg, tr in pair_daughters:
        if m_id not in mothers:
            mothers[m_id] = []
        mothers[m_id].append((i, pdg, tr))

    best_m_id = -1
    best_daughters = []
    for m_id, ds in mothers.items():
        if m_id == -1:
            continue
        has_m = any(d[1] == 13 for d in ds)
        has_p = any(d[1] == -13 for d in ds)
        if has_m and has_p:
            best_m_id = m_id
            mu_m = [d for d in ds if d[1] == 13][0]
            mu_p = [d for d in ds if d[1] == -13][0]
            best_daughters = [mu_m, mu_p]
            break

    # Fallback: check primary muon with daughter muons
    if not best_daughters and primary_idx != -1:
        p_ds = [d for d in pair_daughters if d[1] == primary_idx]
        has_m = any(d[2] == 13 for d in p_ds)
        has_p = any(d[2] == -13 for d in p_ds)
        if has_m and has_p:
            best_m_id = primary_idx
            best_daughters = [
                (p_ds[0][0], p_ds[0][2], p_ds[0][3]),
                (p_ds[1][0], p_ds[1][2], p_ds[1][3])
            ]

    if not best_daughters:
        return {
            'primary_idx': primary_idx,
            'primary_track': primary_track,
            'intermediate_tracks': [],
            'daughter_tracks': [],
            'ancestry_indices': set([primary_idx] if primary_idx != -1 else []),
            'process_type': "Standard MC Event",
            'process_category': "other",
            'z_vertex': 250.0
        }

    # 3. Trace unbroken ancestry chain from immediate mother up to primary
    ancestry_chain = []
    curr_id = best_m_id
    visited = set()
    while curr_id != -1 and curr_id < len(tracks) and curr_id not in visited:
        visited.add(curr_id)
        ancestry_chain.append((curr_id, tracks[curr_id]))
        curr_id = tracks[curr_id].GetMotherId()

    ancestry_chain.reverse() # Starts at primary and ends at immediate mother

    if ancestry_chain and primary_idx == -1:
        primary_idx = ancestry_chain[0][0]
        primary_track = ancestry_chain[0][1]

    immediate_mother_idx, immediate_mother_track = ancestry_chain[-1] if ancestry_chain else (-1, None)
    raw_intermediates = ancestry_chain[1:] if len(ancestry_chain) > 1 else []

    # Classify process
    proc_name = "Direct Muon Trident"
    proc_cat = "direct"
    if immediate_mother_track:
        m_pdg = immediate_mother_track.GetPdgCode()
        if abs(m_pdg) == 13:
            if immediate_mother_idx == primary_idx:
                proc_name = "Direct Muon Trident: #mu #rightarrow #mu + #mu^{+}#mu^{-}"
                proc_cat = "direct"
            else:
                proc_name = "Secondary Muon Trident: #mu_{sec} #rightarrow #mu + #mu^{+}#mu^{-}"
                proc_cat = "secondary_muon"
        elif m_pdg == 22:
            proc_name = "Gamma Conversion: #gamma #rightarrow #mu^{+}#mu^{-}"
            proc_cat = "gamma_conv"
        elif abs(m_pdg) == 11:
            proc_name = "Positron Annihilation: e^{+}e^{-} #rightarrow #mu^{+}#mu^{-}"
            proc_cat = "positron_annihil"
        elif abs(m_pdg) in [111, 211, 321]:
            proc_name = f"Hadronic Trident (Mother PDG {m_pdg})"
            proc_cat = "hadronic"
        else:
            proc_name = f"Trident via Mother PDG {m_pdg}"
            proc_cat = "other"

    # Style intermediate tracks
    intermediate_tracks = []
    for tr_idx, tr_obj in raw_intermediates:
        pdg = tr_obj.GetPdgCode()
        proc = tr_obj.GetProcID()
        if abs(pdg) == 13:
            label = f"Radiating #mu^{{{'-' if pdg==13 else '+'}}}"
            col = ROOT.kMagenta + 2
            style = 1
            width = 3
        elif pdg == 22:
            label = "Intermediate #gamma (Brem)"
            col = ROOT.kOrange + 7
            style = 7
            width = 2
        elif abs(pdg) == 11:
            label = f"Intermediate e^{{{'-' if pdg==11 else '+'}}}"
            col = ROOT.kViolet + 1
            style = 1
            width = 2
        elif abs(pdg) in [211, 111, 321, 2212, 2112]:
            label = f"Intermediate Hadron (PDG {pdg})"
            col = ROOT.kTeal + 2
            style = 1
            width = 2
        else:
            label = f"Intermediate Track (PDG {pdg})"
            col = ROOT.kCyan + 2
            style = 1
            width = 2
        intermediate_tracks.append((tr_idx, tr_obj, label, col, style, width))

    ancestry_indices = set([primary_idx] + [c[0] for c in raw_intermediates] + [d[0] for d in best_daughters])
    z_vert = best_daughters[0][2].GetStartZ() if best_daughters else 250.0

    return {
        'primary_idx': primary_idx,
        'primary_track': primary_track,
        'intermediate_tracks': intermediate_tracks,
        'daughter_tracks': [(d[0], d[1], d[2]) for d in best_daughters], # (idx, pdg, track)
        'ancestry_indices': ancestry_indices,
        'process_type': proc_name,
        'process_category': proc_cat,
        'z_vertex': z_vert
    }

def draw_mctracks(event, topo, geo_ctx, z_range, show_all_mctracks=False, mc_points_map=None):
    """
    Draws the full physical MCTrack path from the initial particle to the muon pair,
    including all in-between intermediate tracks and optional subtle background tracks.
    """
    c_xz = geo_ctx['c_xz']
    c_yz = geo_ctx['c_yz']
    c_3d = geo_ctx['c_3d']
    legend_xz = geo_ctx['legend_xz']
    legend_yz = geo_ctx['legend_yz']

    z_min_2d_plot, z_max_2d_plot = z_range

    def make_array(vals):
        return array.array('d', vals)

    primary_idx = topo['primary_idx']
    primary_track = topo['primary_track']
    intermediate_tracks = topo['intermediate_tracks']
    daughter_tracks = topo['daughter_tracks']
    ancestry_indices = topo['ancestry_indices']

    # 1. Background Tracks (subtle transparent lines)
    gr_other_xz = []
    gr_other_yz = []
    if show_all_mctracks and hasattr(event, "MCTrack"):
        other_trajs_2d = []
        other_trajs_3d = []
        for i_tr, tr in enumerate(event.MCTrack):
            if i_tr in ancestry_indices:
                continue
            x_o_2d, y_o_2d, z_o_2d = get_track_trajectory_points(i_tr, tr, z_min_2d_plot, z_max_2d_plot, mc_points_map)
            if len(z_o_2d) >= 2:
                other_trajs_2d.append((x_o_2d, y_o_2d, z_o_2d))
            x_o_3d, y_o_3d, z_o_3d = get_track_trajectory_points(i_tr, tr, Z_3D_MIN, Z_MAX, mc_points_map)
            if len(z_o_3d) >= 2:
                other_trajs_3d.append((x_o_3d, y_o_3d, z_o_3d))

        c_xz.cd()
        for x_ends, y_ends, z_ends in other_trajs_2d:
            gr_oth = ROOT.TGraph(len(z_ends), make_array(z_ends), make_array(x_ends))
            ROOT.SetOwnership(gr_oth, False)
            gr_oth.SetLineColorAlpha(ROOT.kGray+1, 0.25)
            gr_oth.SetLineWidth(1)
            gr_oth.Draw("L")
            gr_other_xz.append(gr_oth)

        c_yz.cd()
        for x_ends, y_ends, z_ends in other_trajs_2d:
            gr_oth = ROOT.TGraph(len(z_ends), make_array(z_ends), make_array(y_ends))
            ROOT.SetOwnership(gr_oth, False)
            gr_oth.SetLineColorAlpha(ROOT.kGray+1, 0.25)
            gr_oth.SetLineWidth(1)
            gr_oth.Draw("L")
            gr_other_yz.append(gr_oth)

        c_3d.cd()
        for x_ends, y_ends, z_ends in other_trajs_3d:
            pl_oth = ROOT.TPolyLine3D(len(z_ends))
            ROOT.SetOwnership(pl_oth, False)
            for idx_p in range(len(z_ends)):
                pl_oth.SetPoint(idx_p, z_ends[idx_p], x_ends[idx_p], y_ends[idx_p])
            pl_oth.SetLineColorAlpha(ROOT.kGray+1, 0.20)
            pl_oth.SetLineWidth(1)
            pl_oth.Draw("same")

    # 2. Intermediate Ancestor Tracks (In-between tracks!)
    gr_inter_xz = []
    gr_inter_yz = []
    for tr_idx, tr_obj, label, col, style, width in intermediate_tracks:
        x_i_2d, y_i_2d, z_i_2d = get_track_trajectory_points(tr_idx, tr_obj, z_min_2d_plot, z_max_2d_plot, mc_points_map)
        if len(z_i_2d) >= 2:
            c_xz.cd()
            gr_ix = ROOT.TGraph(len(z_i_2d), make_array(z_i_2d), make_array(x_i_2d))
            ROOT.SetOwnership(gr_ix, False)
            gr_ix.SetLineColor(col)
            gr_ix.SetLineStyle(style)
            gr_ix.SetLineWidth(width)
            gr_ix.Draw("L")
            gr_inter_xz.append((gr_ix, label))

            c_yz.cd()
            gr_iy = ROOT.TGraph(len(z_i_2d), make_array(z_i_2d), make_array(y_i_2d))
            ROOT.SetOwnership(gr_iy, False)
            gr_iy.SetLineColor(col)
            gr_iy.SetLineStyle(style)
            gr_iy.SetLineWidth(width)
            gr_iy.Draw("L")
            gr_inter_yz.append((gr_iy, label))

            c_3d.cd()
            x_i_3d, y_i_3d, z_i_3d = get_track_trajectory_points(tr_idx, tr_obj, Z_3D_MIN, Z_MAX, mc_points_map)
            if len(z_i_3d) >= 2:
                pl_i_3d = ROOT.TPolyLine3D(len(z_i_3d))
                ROOT.SetOwnership(pl_i_3d, False)
                for idx_p in range(len(z_i_3d)):
                    pl_i_3d.SetPoint(idx_p, z_i_3d[idx_p], x_i_3d[idx_p], y_i_3d[idx_p])
                pl_i_3d.SetLineColor(col)
                pl_i_3d.SetLineStyle(style)
                pl_i_3d.SetLineWidth(width)
                pl_i_3d.Draw("same")

    # 3. Primary Track
    gr_p_xz, gr_p_yz, pl_p_3d = None, None, None
    if primary_track:
        x_p, y_p, z_p = get_track_trajectory_points(primary_idx, primary_track, z_min_2d_plot, z_max_2d_plot, mc_points_map)
        if len(z_p) >= 2:
            p_pdg = primary_track.GetPdgCode()
            p_label = f"Primary #mu^{{{'-' if p_pdg==13 else '+'}}}" if abs(p_pdg) == 13 else f"Primary Track (PDG {p_pdg})"
            c_xz.cd()
            gr_p_xz = ROOT.TGraph(len(z_p), make_array(z_p), make_array(x_p))
            ROOT.SetOwnership(gr_p_xz, False)
            gr_p_xz.SetLineColor(ROOT.kBlack)
            gr_p_xz.SetLineWidth(3)
            gr_p_xz.Draw("L")
            legend_xz.AddEntry(gr_p_xz, p_label, "l")

            c_yz.cd()
            gr_p_yz = ROOT.TGraph(len(z_p), make_array(z_p), make_array(y_p))
            ROOT.SetOwnership(gr_p_yz, False)
            gr_p_yz.SetLineColor(ROOT.kBlack)
            gr_p_yz.SetLineWidth(3)
            gr_p_yz.Draw("L")
            legend_yz.AddEntry(gr_p_yz, p_label, "l")

            c_3d.cd()
            x_p_3d, y_p_3d, z_p_3d = get_track_trajectory_points(primary_idx, primary_track, Z_3D_MIN, Z_MAX, mc_points_map)
            if len(z_p_3d) >= 2:
                pl_p_3d = ROOT.TPolyLine3D(len(z_p_3d))
                ROOT.SetOwnership(pl_p_3d, False)
                for idx_p in range(len(z_p_3d)):
                    pl_p_3d.SetPoint(idx_p, z_p_3d[idx_p], x_p_3d[idx_p], y_p_3d[idx_p])
                pl_p_3d.SetLineColor(ROOT.kBlack)
                pl_p_3d.SetLineWidth(3)
                pl_p_3d.Draw("same")

    # Add intermediate entries to legend (deduplicated by label)
    added_labels_xz = set()
    for gr, label in gr_inter_xz:
        if label not in added_labels_xz:
            legend_xz.AddEntry(gr, label, "l")
            added_labels_xz.add(label)

    added_labels_yz = set()
    for gr, label in gr_inter_yz:
        if label not in added_labels_yz:
            legend_yz.AddEntry(gr, label, "l")
            added_labels_yz.add(label)

    # 4. Daughter Tracks
    gr_ds_xz = []
    gr_ds_yz = []
    pls_3d = []
    for d_i, d_pdg, d_track in daughter_tracks:
        x_d, y_d, z_d = get_track_trajectory_points(d_i, d_track, z_min_2d_plot, z_max_2d_plot, mc_points_map)
        if len(z_d) < 2:
            continue
        c_xz.cd()
        gr_x = ROOT.TGraph(len(z_d), make_array(z_d), make_array(x_d))
        ROOT.SetOwnership(gr_x, False)
        gr_x.SetLineWidth(3)
        gr_x.SetLineStyle(2)
        gr_x.SetLineColor(ROOT.kBlue if d_pdg == 13 else ROOT.kRed)
        gr_x.Draw("L")
        gr_ds_xz.append((gr_x, d_pdg))

        c_yz.cd()
        gr_y = ROOT.TGraph(len(z_d), make_array(z_d), make_array(y_d))
        ROOT.SetOwnership(gr_y, False)
        gr_y.SetLineWidth(3)
        gr_y.SetLineStyle(2)
        gr_y.SetLineColor(ROOT.kBlue if d_pdg == 13 else ROOT.kRed)
        gr_y.Draw("L")
        gr_ds_yz.append((gr_y, d_pdg))

        c_3d.cd()
        x_d_3d, y_d_3d, z_d_3d = get_track_trajectory_points(d_i, d_track, Z_3D_MIN, Z_MAX, mc_points_map)
        if len(z_d_3d) >= 2:
            pl_3 = ROOT.TPolyLine3D(len(z_d_3d))
            ROOT.SetOwnership(pl_3, False)
            for idx_p in range(len(z_d_3d)):
                pl_3.SetPoint(idx_p, z_d_3d[idx_p], x_d_3d[idx_p], y_d_3d[idx_p])
            pl_3.SetLineColor(ROOT.kBlue if d_pdg == 13 else ROOT.kRed)
            pl_3.SetLineStyle(2)
            pl_3.SetLineWidth(3)
            pl_3.Draw("same")
            pls_3d.append(pl_3)

    if gr_ds_xz:
        legend_xz.AddEntry(gr_ds_xz[0][0], "Trident #mu^{-} / #mu^{+} daughters", "l")
        legend_yz.AddEntry(gr_ds_yz[0][0], "Trident #mu^{-} / #mu^{+} daughters", "l")

    if show_all_mctracks and gr_other_xz:
        legend_xz.AddEntry(gr_other_xz[0], "Other MC Tracks", "l")
        legend_yz.AddEntry(gr_other_yz[0], "Other MC Tracks", "l")

    return {
        'gr_p_xz': gr_p_xz, 'gr_p_yz': gr_p_yz, 'pl_p_3d': pl_p_3d,
        'gr_inter_xz': gr_inter_xz, 'gr_inter_yz': gr_inter_yz,
        'gr_ds_xz': gr_ds_xz, 'gr_ds_yz': gr_ds_yz, 'pls_3d': pls_3d,
        'gr_other_xz': gr_other_xz, 'gr_other_yz': gr_other_yz
    }

def extract_hough_lines(event, muon_reco_task=None, snd_geo=None, run_reco_if_missing=False,
                        scifi_lines_z_range=(260.0, 375.0), ds_lines_z_range=(375.0, 560.0), **kwargs):
    """
    Extracts reconstructed Hough transform lines for SciFi and/or Downstream (DS) MuFilter.
    1. First checks if Hough parameter branches already exist in the event:
       - SciFi: xz_m1, xz_c1, yz_m1, yz_c1, ..., xz_m3, yz_c3 (and variations)
       - DS: xz_ds_m1, xz_ds_c1, ds_xz_m1, ds_xz_c1, xz_m1_ds, xz_c1_ds, etc.
    2. If branches are missing/empty and run_reco_if_missing is True with initialized reco task,
       runs Hough transform on the fly via SndlhcMuonReco.
    Returns:
      hough_lines = {
          'scifi': {
              'XZ': [(line_idx, slope, intercept, z1, z2), ...],
              'YZ': [(line_idx, slope, intercept, z1, z2), ...]
          },
          'ds': {
              'XZ': [(line_idx, slope, intercept, z1, z2), ...],
              'YZ': [(line_idx, slope, intercept, z1, z2), ...]
          }
      }
    """
    if 'scifi_trk_z_range' in kwargs:
        scifi_lines_z_range = kwargs['scifi_trk_z_range']
    if 'scifi_z_range' in kwargs:
        scifi_lines_z_range = kwargs['scifi_z_range']
    if 'ds_trk_z_range' in kwargs:
        ds_lines_z_range = kwargs['ds_trk_z_range']
    if 'ds_z_range' in kwargs:
        ds_lines_z_range = kwargs['ds_z_range']

    import math

    def is_valid(val):
        if val is None:
            return False
        try:
            f = float(val)
            return not (math.isnan(f) or math.isinf(f) or f in [-999.0, -1.0e9, 999.0])
        except (ValueError, TypeError):
            return False

    scifi_lines = {'XZ': [], 'YZ': []}
    ds_lines = {'XZ': [], 'YZ': []}

    z1_sf, z2_sf = scifi_lines_z_range
    z1_ds, z2_ds = ds_lines_z_range

    # 1. Scan tree branches for SciFi Hough lines
    for idx in range(1, 10):
        # SciFi XZ
        candidates_xz_m = [f"xz_sf_m{idx}", f"xz_m{idx}", f"xz_m_{idx}", f"scifi_xz_m{idx}", f"scifi_xz_m_{idx}", f"hough_xz_m{idx}"]
        candidates_xz_c = [f"xz_sf_c{idx}", f"xz_c{idx}", f"xz_c_{idx}", f"scifi_xz_c{idx}", f"scifi_xz_c_{idx}", f"hough_xz_c{idx}"]
        mx = next((getattr(event, n) for n in candidates_xz_m if hasattr(event, n) and is_valid(getattr(event, n))), None)
        cx = next((getattr(event, n) for n in candidates_xz_c if hasattr(event, n) and is_valid(getattr(event, n))), None)
        if mx is not None and cx is not None:
            scifi_lines['XZ'].append((idx, float(mx), float(cx), z1_sf, z2_sf))

        # SciFi YZ
        candidates_yz_m = [f"yz_sf_m{idx}", f"yz_m{idx}", f"yz_m_{idx}", f"scifi_yz_m{idx}", f"scifi_yz_m_{idx}", f"hough_yz_m{idx}"]
        candidates_yz_c = [f"yz_sf_c{idx}", f"yz_c{idx}", f"yz_c_{idx}", f"scifi_yz_c{idx}", f"scifi_yz_c_{idx}", f"hough_yz_c{idx}"]
        my = next((getattr(event, n) for n in candidates_yz_m if hasattr(event, n) and is_valid(getattr(event, n))), None)
        cy = next((getattr(event, n) for n in candidates_yz_c if hasattr(event, n) and is_valid(getattr(event, n))), None)
        if my is not None and cy is not None:
            scifi_lines['YZ'].append((idx, float(my), float(cy), z1_sf, z2_sf))

    # 2. Scan tree branches for Downstream (DS) Hough lines
    for idx in range(1, 10):
        # DS XZ
        candidates_ds_xz_m = [f"xz_ds_m{idx}", f"ds_xz_m{idx}", f"xz_m{idx}_ds", f"xz_m_{idx}_ds", f"ds_xz_m_{idx}"]
        candidates_ds_xz_c = [f"xz_ds_c{idx}", f"ds_xz_c{idx}", f"xz_c{idx}_ds", f"xz_c_{idx}_ds", f"ds_xz_c_{idx}"]
        mx_ds = next((getattr(event, n) for n in candidates_ds_xz_m if hasattr(event, n) and is_valid(getattr(event, n))), None)
        cx_ds = next((getattr(event, n) for n in candidates_ds_xz_c if hasattr(event, n) and is_valid(getattr(event, n))), None)
        if mx_ds is not None and cx_ds is not None:
            ds_lines['XZ'].append((idx, float(mx_ds), float(cx_ds), z1_ds, z2_ds))

        # DS YZ
        candidates_ds_yz_m = [f"yz_ds_m{idx}", f"ds_yz_m{idx}", f"yz_m{idx}_ds", f"yz_m_{idx}_ds", f"ds_yz_m_{idx}"]
        candidates_ds_yz_c = [f"yz_ds_c{idx}", f"ds_yz_c{idx}", f"yz_c{idx}_ds", f"yz_c_{idx}_ds", f"ds_yz_c_{idx}"]
        my_ds = next((getattr(event, n) for n in candidates_ds_yz_m if hasattr(event, n) and is_valid(getattr(event, n))), None)
        cy_ds = next((getattr(event, n) for n in candidates_ds_yz_c if hasattr(event, n) and is_valid(getattr(event, n))), None)
        if my_ds is not None and cy_ds is not None:
            ds_lines['YZ'].append((idx, float(my_ds), float(cy_ds), z1_ds, z2_ds))

    # 3. Optional on-the-fly reconstruction fallback
    if run_reco_if_missing and muon_reco_task and snd_geo and not (scifi_lines['XZ'] or scifi_lines['YZ'] or ds_lines['XZ'] or ds_lines['YZ']):
        try:
            import SndlhcMuonReco
            import numpy as np

            def run_quick_hough(system_name):
                hit_col = {"pos": [[], [], []], "d": [[], [], []], "vert": [], "system": [], "detectorID": []}
                pos_a, pos_b = ROOT.TVector3(), ROOT.TVector3()

                if system_name in ["scifi", "both"]:
                    scifi_hits = getattr(event, "Digi_ScifiHits", None)
                    if scifi_hits:
                        for h in scifi_hits:
                            if not h.isValid(): continue
                            snd_geo.modules["Scifi"].GetSiPMPosition(h.GetDetectorID(), pos_a, pos_b)
                            for ax in range(3): hit_col["pos"][ax].append(pos_a[ax])
                            hit_col["d"][0].append(muon_reco_task.Scifi_dx)
                            hit_col["d"][1].append(muon_reco_task.Scifi_dy)
                            hit_col["d"][2].append(muon_reco_task.Scifi_dz)
                            hit_col["vert"].append(h.isVertical())
                            hit_col["system"].append(0)
                            hit_col["detectorID"].append(h.GetDetectorID())

                if system_name in ["ds", "both"]:
                    mufi_hits = getattr(event, "Digi_MuFilterHits", None)
                    if mufi_hits:
                        for h in mufi_hits:
                            if not h.isValid() or h.GetSystem() != 3: continue
                            snd_geo.modules["MuFilter"].GetPosition(h.GetDetectorID(), pos_a, pos_b)
                            for ax in range(3): hit_col["pos"][ax].append(pos_a[ax])
                            hit_col["d"][0].append(muon_reco_task.MuFilter_ds_dx)
                            hit_col["d"][1].append(muon_reco_task.MuFilter_ds_dx)
                            hit_col["d"][2].append(muon_reco_task.MuFilter_ds_dz)
                            hit_col["vert"].append(h.isVertical())
                            hit_col["system"].append(h.GetSystem())
                            hit_col["detectorID"].append(h.GetDetectorID())

                if not hit_col["pos"][0]:
                    return {'XZ': [], 'YZ': []}

                for k in ["pos", "d"]: hit_col[k] = np.array(hit_col[k], dtype=np.float32)
                for k, dt in [("vert", np.bool_), ("system", np.int32), ("detectorID", np.int32)]:
                    hit_col[k] = np.array(hit_col[k], dtype=dt)

                res_lines = {'XZ': [], 'YZ': []}
                for proj in ['XZ', 'YZ']:
                    is_vert = (proj == 'XZ')
                    h_obj = muon_reco_task.h_ZX if is_vert else muon_reco_task.h_ZY
                    axis = 0 if is_vert else 1
                    mask = (hit_col['vert'] == is_vert)
                    if not np.any(mask): continue

                    fit = h_obj.fit_randomize(
                        np.dstack([hit_col['pos'][2][mask], hit_col['pos'][axis][mask]])[0],
                        np.dstack([hit_col['d'][2][mask], hit_col['d'][axis][mask]])[0],
                        muon_reco_task.n_random, False, False
                    )
                    if fit[0] not in [-1, -999]:
                        z1_val = z1_sf if system_name == "scifi" else z1_ds
                        z2_val = z2_sf if system_name == "scifi" else z2_ds
                        res_lines[proj].append((1, float(fit[0]), float(fit[1]), z1_val, z2_val))
                return res_lines

            sf_res = run_quick_hough("scifi")
            scifi_lines['XZ'].extend(sf_res['XZ'])
            scifi_lines['YZ'].extend(sf_res['YZ'])

            ds_res = run_quick_hough("ds")
            ds_lines['XZ'].extend(ds_res['XZ'])
            ds_lines['YZ'].extend(ds_res['YZ'])

        except Exception as e:
            print(f"Warning: Hough reconstruction fallback encountered an error: {e}")

    return {
        'scifi': scifi_lines,
        'ds': ds_lines
    }

def draw_hough_lines(event, hough_lines, geo_ctx):
    """
    Draws reconstructed Hough transform lines on top of 2D projection and 3D canvases.
    """
    c_xz = geo_ctx['c_xz']
    c_yz = geo_ctx['c_yz']
    c_3d = geo_ctx['c_3d']
    legend_xz = geo_ctx['legend_xz']
    legend_yz = geo_ctx['legend_yz']

    gr_hough_xz = []
    gr_hough_yz = []
    poly3d_hough = []

    # High-contrast distinctive palettes for Hough lines
    scifi_palette = [ROOT.kCyan+2, ROOT.kOrange+2, ROOT.kViolet+2]
    ds_palette = [ROOT.kGreen+2, ROOT.kSpring-5]

    # 1. SciFi Hough Lines
    scifi_xz = hough_lines.get('scifi', {}).get('XZ', [])
    scifi_yz = hough_lines.get('scifi', {}).get('YZ', [])

    c_xz.cd()
    for l_idx, m, c, z1, z2 in scifi_xz:
        color = scifi_palette[0]
        gr = ROOT.TGraph(2)
        ROOT.SetOwnership(gr, False)
        gr.SetPoint(0, z1, m * z1 + c)
        gr.SetPoint(1, z2, m * z2 + c)
        gr.SetLineColor(color)
        gr.SetLineStyle(1)
        gr.SetLineWidth(3)
        gr.Draw("L same")
        label = "Hough Line (SciFi)"
        gr_hough_xz.append((gr, label))

    c_yz.cd()
    for l_idx, m, c, z1, z2 in scifi_yz:
        color = scifi_palette[0]
        gr = ROOT.TGraph(2)
        ROOT.SetOwnership(gr, False)
        gr.SetPoint(0, z1, m * z1 + c)
        gr.SetPoint(1, z2, m * z2 + c)
        gr.SetLineColor(color)
        gr.SetLineStyle(1)
        gr.SetLineWidth(3)
        gr.Draw("L same")
        label = "Hough Line (SciFi)"
        gr_hough_yz.append((gr, label))

    # 2. Downstream (DS) Hough Lines
    ds_xz = hough_lines.get('ds', {}).get('XZ', [])
    ds_yz = hough_lines.get('ds', {}).get('YZ', [])

    c_xz.cd()
    for l_idx, m, c, z1, z2 in ds_xz:
        color = ds_palette[0]
        gr = ROOT.TGraph(2)
        ROOT.SetOwnership(gr, False)
        gr.SetPoint(0, z1, m * z1 + c)
        gr.SetPoint(1, z2, m * z2 + c)
        gr.SetLineColor(color)
        gr.SetLineStyle(1)
        gr.SetLineWidth(3)
        gr.Draw("L same")
        label = "Hough Line (DS)"
        gr_hough_xz.append((gr, label))

    c_yz.cd()
    for l_idx, m, c, z1, z2 in ds_yz:
        color = ds_palette[0]
        gr = ROOT.TGraph(2)
        ROOT.SetOwnership(gr, False)
        gr.SetPoint(0, z1, m * z1 + c)
        gr.SetPoint(1, z2, m * z2 + c)
        gr.SetLineColor(color)
        gr.SetLineStyle(1)
        gr.SetLineWidth(3)
        gr.Draw("L same")
        label = "Hough Line (DS)"
        gr_hough_yz.append((gr, label))

    # Add to legend (deduplicated)
    added_labels_xz = set()
    for gr, label in gr_hough_xz:
        if label not in added_labels_xz:
            legend_xz.AddEntry(gr, label, "l")
            added_labels_xz.add(label)

    added_labels_yz = set()
    for gr, label in gr_hough_yz:
        if label not in added_labels_yz:
            legend_yz.AddEntry(gr, label, "l")
            added_labels_yz.add(label)

    return {
        'gr_hough_xz': gr_hough_xz,
        'gr_hough_yz': gr_hough_yz,
        'poly3d_hough': poly3d_hough
    }

def create_event_display(event, evt_num, i_event, geo_elements, ggeo, scifi_module, mufilter_module, snd_geo,
                          topo, file_tag="", show_all_mctracks=False, draw_hits=True, draw_tracks=True,
                          draw_hough=True, hough_lines=None,
                          canvas_z_range=(250.0, 600.0), start_from_primary_muon=False,
                          scifi_lines_z_range=(260.0, 375.0), ds_lines_z_range=(375.0, 560.0), **kwargs):
    """
    High-level orchestrator that creates canvases, calls draw_detector_geometry,
    draw_detector_hits, draw_mctracks, and draw_hough_lines, draws legends/banners, and returns the canvases.
    """
    if 'scifi_trk_z_range' in kwargs:
        scifi_lines_z_range = kwargs['scifi_trk_z_range']
    if 'scifi_z_range' in kwargs:
        scifi_lines_z_range = kwargs['scifi_z_range']
    if 'ds_trk_z_range' in kwargs:
        ds_lines_z_range = kwargs['ds_trk_z_range']
    if 'ds_z_range' in kwargs:
        ds_lines_z_range = kwargs['ds_z_range']

    mc_points_map = build_mc_points_map(event)

    primary_idx = topo['primary_idx']
    primary_track = topo['primary_track']
    intermediate_tracks = topo['intermediate_tracks']
    daughter_tracks = topo['daughter_tracks']
    process_type = topo['process_type']
    z_vert_val = topo['z_vertex']

    if start_from_primary_muon and primary_track:
        z_min_2d = primary_track.GetStartZ()
    else:
        z_min_2d = canvas_z_range[0]
    z_max_2d = canvas_z_range[1]

    x_p, y_p, z_p = get_track_trajectory_points(primary_idx, primary_track, z_min_2d, z_max_2d, mc_points_map) if primary_track else ([], [], [])

    inter_trajs = []
    for tr_idx, tr_obj, label, col, style, width in intermediate_tracks:
        x_i, y_i, z_i = get_track_trajectory_points(tr_idx, tr_obj, z_min_2d, z_max_2d, mc_points_map)
        inter_trajs.append((x_i, y_i, z_i))

    d_trajs = []
    for d_i, d_pdg, d_track in daughter_tracks:
        d_z_start = max(d_track.GetStartZ(), z_min_2d)
        x_d, y_d, z_d = get_track_trajectory_points(d_i, d_track, d_z_start, z_max_2d, mc_points_map)
        d_trajs.append((d_pdg, x_d, y_d, z_d))

    # Filter visible geometry elements in Z
    visible_geo = [elem for elem in geo_elements if elem['z'][1] >= z_min_2d and elem['z'][0] <= z_max_2d]
    if not visible_geo:
        visible_geo = geo_elements

    all_x = list(x_p) + [x for d in d_trajs for x in d[1]] + [x for it in inter_trajs for x in it[0]] + [elem['x'][0] for elem in visible_geo] + [elem['x'][1] for elem in visible_geo]
    all_y = list(y_p) + [y for d in d_trajs for y in d[2]] + [y for it in inter_trajs for y in it[1]] + [elem['y'][0] for elem in visible_geo] + [elem['y'][1] for elem in visible_geo]

    x_min_plot = (min(all_x) - 5) if all_x else -85.0
    x_max_plot = (max(all_x) + 5) if all_x else 15.0
    y_min_plot = (min(all_y) - 5) if all_y else -15.0
    y_max_plot = (max(all_y) + 5) if all_y else 85.0
    z_min_2d_plot, z_max_2d_plot = z_min_2d, z_max_2d

    proc_tag = "_" + topo['process_category'] if topo['process_category'] != "direct" else ""
    evt_dir_name = f"Event_{evt_num}_Zvert{int(z_vert_val)}cm{proc_tag}"

    # 1. Draw Detector Geometry
    geo_ctx = draw_detector_geometry(
        geo_elements,
        z_range=(z_min_2d_plot, z_max_2d_plot),
        x_range=(x_min_plot, x_max_plot),
        y_range=(y_min_plot, y_max_plot),
        ggeo=ggeo, evt_num=evt_num, i_event=i_event
    )

    # 2. Draw Detector Hits
    hits_ctx = {}
    if draw_hits:
        hits_ctx = draw_detector_hits(event, scifi_module, mufilter_module, geo_ctx, snd_geo=snd_geo)

    # 3. Draw MCTracks (Full Ancestry Path)
    mctrack_ctx = {}
    if draw_tracks:
        mctrack_ctx = draw_mctracks(
            event, topo, geo_ctx, z_range=(z_min_2d_plot, z_max_2d_plot),
            show_all_mctracks=show_all_mctracks, mc_points_map=mc_points_map
        )

    # 4. Draw Hough Reconstructed Lines (On top of everything else!)
    hough_ctx = {}
    if draw_hough:
        if hough_lines is None:
            hough_lines = extract_hough_lines(event, scifi_lines_z_range=scifi_lines_z_range, ds_lines_z_range=ds_lines_z_range)
        hough_ctx = draw_hough_lines(event, hough_lines, geo_ctx)

    # 5. Draw Legends and Banners
    c_xz = geo_ctx['c_xz']
    c_yz = geo_ctx['c_yz']
    c_3d = geo_ctx['c_3d']

    legend_xz = geo_ctx['legend_xz']
    legend_yz = geo_ctx['legend_yz']

    # Update legends
    seen_labels = set()
    for tr_i, label, col, style, width in mctrack_ctx.get('drawn_intermediate', []):
        if label not in seen_labels:
            dummy_gr = ROOT.TGraph()
            ROOT.SetOwnership(dummy_gr, False)
            dummy_gr.SetLineColor(col)
            dummy_gr.SetLineStyle(style)
            dummy_gr.SetLineWidth(width)
            legend_xz.AddEntry(dummy_gr, label, "l")
            legend_yz.AddEntry(dummy_gr, label, "l")
            seen_labels.add(label)

    for d_pdg, label, col, style, width in mctrack_ctx.get('drawn_daughters', []):
        if label not in seen_labels:
            dummy_gr = ROOT.TGraph()
            ROOT.SetOwnership(dummy_gr, False)
            dummy_gr.SetLineColor(col)
            dummy_gr.SetLineStyle(style)
            dummy_gr.SetLineWidth(width)
            legend_xz.AddEntry(dummy_gr, label, "l")
            legend_yz.AddEntry(dummy_gr, label, "l")
            seen_labels.add(label)

    c_xz.cd()
    legend_xz.Draw()
    c_yz.cd()
    legend_yz.Draw()

    banner_event = f"Event #{evt_num}  (Entry {i_event})"
    banner_process = f"Process: {process_type} | Z_vtx = {z_vert_val:.1f} cm"

    c_xz.cd()
    t_lat = ROOT.TLatex()
    ROOT.SetOwnership(t_lat, False)
    t_lat.SetNDC()
    t_lat.SetTextFont(42)
    t_lat.SetTextSize(0.033)
    t_lat.DrawLatex(0.12, 0.94, banner_event)
    t_lat.SetTextSize(0.024)
    t_lat.DrawLatex(0.12, 0.905, banner_process)

    c_yz.cd()
    t_lat_y = ROOT.TLatex()
    ROOT.SetOwnership(t_lat_y, False)
    t_lat_y.SetNDC()
    t_lat_y.SetTextFont(42)
    t_lat_y.SetTextSize(0.033)
    t_lat_y.DrawLatex(0.12, 0.94, banner_event)
    t_lat_y.SetTextSize(0.024)
    t_lat_y.DrawLatex(0.12, 0.905, banner_process)

    c_3d.cd()
    banner_3d = ROOT.TLatex()
    ROOT.SetOwnership(banner_3d, False)
    banner_3d.SetNDC()
    banner_3d.SetTextFont(42)
    banner_3d.SetTextSize(0.03)
    banner_3d.DrawLatex(0.03, 0.94, banner_event)

    c_xz.Update()
    c_yz.Update()
    c_3d.Update()

    return c_xz, c_yz, c_3d, evt_dir_name, (topo['process_category'] != 'direct'), z_vert_val

def process_single_file(input_file, output_file, geo_elements, ggeo, scifi_module, mufilter_module, snd_geo,
                        max_displays=0, show_all_mctracks=False, draw_hough=True,
                        canvas_z_range=(250.0, 600.0), start_from_primary_muon=False,
                        scifi_lines_z_range=(260.0, 375.0), ds_lines_z_range=(375.0, 560.0),
                        target_region_z_range=(260.0, 355.0), save_trees=False, **kwargs):
    """
    Processes events in the input file without filtering:
    - Generates event displays for every event (or up to max_displays).
    - Optionally displays Hough reconstructed lines on top of detector hits and MCTracks.
    - Stores event displays inside the output ROOT file organized by process category and detector region:
      EventDisplays/<Process_Category>/<Region_Category>/<Event_Dir>/
    - If save_trees=True, also includes the original 'cbmsim' tree and metadata in the output ROOT file.
    """
    if 'scifi_trk_z_range' in kwargs:
        scifi_lines_z_range = kwargs['scifi_trk_z_range']
    if 'scifi_z_range' in kwargs:
        scifi_lines_z_range = kwargs['scifi_z_range']
    if 'ds_trk_z_range' in kwargs:
        ds_lines_z_range = kwargs['ds_trk_z_range']
    if 'ds_z_range' in kwargs:
        ds_lines_z_range = kwargs['ds_z_range']
    if 'target_z_range' in kwargs:
        target_region_z_range = kwargs['target_z_range']

    t0 = time.time()

    f_in = None
    if save_trees:
        import shutil
        if os.path.abspath(input_file) != os.path.abspath(output_file):
            shutil.copyfile(input_file, output_file)
        f_out = ROOT.TFile.Open(output_file, "UPDATE")
        tree = f_out.Get("cbmsim") if f_out else None
    else:
        f_in = ROOT.TFile.Open(input_file, "READ")
        tree = f_in.Get("cbmsim") if f_in else None
        f_out = ROOT.TFile.Open(output_file, "RECREATE")

    if not f_out or f_out.IsZombie():
        print(f"Error: Could not open output file '{output_file}'")
        if f_in:
            f_in.Close()
        return 0, 0, []

    if not tree:
        print(f"Error: 'cbmsim' TTree not found in '{input_file}'")
        f_out.Close()
        if f_in:
            f_in.Close()
        return 0, 0, []

    total_events = tree.GetEntries()
    file_tag = extract_file_tag(input_file)

    displays_dir = f_out.Get("EventDisplays")
    if not displays_dir:
        displays_dir = f_out.mkdir("EventDisplays")
    ROOT.SetOwnership(displays_dir, False)

    events_to_process = total_events if (max_displays <= 0 or max_displays > total_events) else max_displays
    events_plotted = 0
    sec_inducing_events = []

    for i_event in range(events_to_process):
        tree.GetEntry(i_event)
        event = tree

        evt_num = None
        if hasattr(event, "EventHeader") and event.EventHeader:
            try:
                evt_num = event.EventHeader.GetEventNumber()
            except Exception:
                pass
        if evt_num is None and hasattr(event, "MCEventHeader") and event.MCEventHeader:
            try:
                evt_num = event.MCEventHeader.GetEventID()
            except Exception:
                pass
        if evt_num is None or evt_num < 0:
            evt_num = i_event

        topo = identify_trident_topology(event)
        hough_lines = extract_hough_lines(
            event,
            scifi_lines_z_range=scifi_lines_z_range,
            ds_lines_z_range=ds_lines_z_range
        ) if draw_hough else None

        c_xz, c_yz, c_3d, evt_dir_name, is_non_direct, z_vert_val = create_event_display(
            event, evt_num, i_event, geo_elements, ggeo, scifi_module, mufilter_module, snd_geo,
            topo=topo, file_tag=file_tag,
            show_all_mctracks=show_all_mctracks, draw_hits=True, draw_tracks=True,
            draw_hough=draw_hough, hough_lines=hough_lines,
            canvas_z_range=canvas_z_range,
            start_from_primary_muon=start_from_primary_muon,
            scifi_lines_z_range=scifi_lines_z_range,
            ds_lines_z_range=ds_lines_z_range
        )

        category_dir_name = {
            'direct': 'Genuine_Tridents',
            'secondary_muon': 'Genuine_Tridents',
            'gamma_conv': 'Gamma_Conversion',
            'positron_annihil': 'Positron_Annihilation',
            'hadronic': 'Hadronic_Tridents',
            'other': 'Other_Events'
        }.get(topo['process_category'], 'Other_Events')

        target_z_min, target_z_max = target_region_z_range
        if z_vert_val < target_z_min:
            region_dir_name = "Upstream_Rock"
        elif z_vert_val < target_z_max:
            region_dir_name = "Target_Region"
        else:
            region_dir_name = "Muon_System"

        cat_dir = displays_dir.Get(category_dir_name)
        if not cat_dir:
            cat_dir = displays_dir.mkdir(category_dir_name)
        ROOT.SetOwnership(cat_dir, False)

        reg_dir = cat_dir.Get(region_dir_name)
        if not reg_dir:
            reg_dir = cat_dir.mkdir(region_dir_name)
        ROOT.SetOwnership(reg_dir, False)

        evt_dir = reg_dir.mkdir(evt_dir_name)
        if not evt_dir:
            evt_dir = reg_dir.Get(evt_dir_name)
        ROOT.SetOwnership(evt_dir, False)
        evt_dir.cd()

        if is_non_direct:
            sec_inducing_events.append((evt_num, i_event, z_vert_val, topo['process_type'], output_file))
            print(f"  [{topo['process_type']}] Event {evt_num} (Entry {i_event}): Z_vtx = {z_vert_val:.1f} cm")

        c_xz.Write("XZ_Projection")
        c_yz.Write("YZ_Projection")
        c_3d.Write("3D_View")
        events_plotted += 1

    f_out.Close()
    if f_in:
        f_in.Close()

    elapsed = time.time() - t0
    print(f"  Processed {events_to_process:,} events in {elapsed:.1f}s | Displays generated: {events_plotted:,}")
    return total_events, events_plotted, sec_inducing_events

def main():
    default_input = "trimuon_filtered_*.root"
    default_output = "trimuon_displays_%s.root"
    default_geofile = "python/geofile_full.Ntuple-TGeant4_boost100.0.root"

    parser = argparse.ArgumentParser(description="Generate Event Displays for Events in Input ROOT Files (No Filtering)")
    parser.add_argument("-i", "--input", dest="input_pattern", default=default_input, help="Input ROOT file path or wildcard pattern (default: %(default)s)")
    parser.add_argument("-g", "--geofile", dest="geofile_filename", default=default_geofile, help="ROOT geometry file")
    parser.add_argument("-o", "--output", dest="output_pattern", default=default_output, help="Output ROOT file path or pattern with %%s for tag (default: %(default)s)")
    parser.add_argument("-n", "--max_displays", dest="max_displays", type=int, default=0, help="Max event displays to generate per file (0 = all events, default: %(default)s)")
    parser.add_argument("-a", "--all_mctracks", "--show_all_mctracks", dest="show_all_mctracks", action="store_true", default=False, help="Include all other MCTracks as thin, transparent grey lines")
    parser.add_argument("--hough", "--draw-hough", dest="draw_hough", action="store_true", default=True, help="Draw Hough transform reconstructed lines if available (default: %(default)s)")
    parser.add_argument("--save-trees", "--include-trees", "--save-tree", dest="save_trees", action="store_true", default=False, help="Include original event tree (cbmsim) and metadata in output ROOT file (default: %(default)s, only canvases saved)")
    parser.add_argument("--z-range", dest="canvas_z_range", nargs=2, type=float, default=[250.0, 600.0], metavar=("Z_MIN", "Z_MAX"), help="Visible Z range for 2D canvas displays in cm (default: 250.0 600.0, covers SciFi and MuFilter)")
    parser.add_argument("--start-from-primary-muon", "--full-primary-z", dest="start_from_primary_muon", action="store_true", default=False, help="Override canvas Z_MIN dynamically to start from the primary incident muon (mother_id == -1) in the upstream rock (default: %(default)s)")
    parser.add_argument("--scifi-lines-z-range", "--scifi-trk-z-range", "--scifi-z-range", dest="scifi_lines_z_range", nargs=2, type=float, default=[260.0, 375.0], metavar=("Z1", "Z2"), help="Z extent range for reconstructed SciFi Hough lines in cm (default: 260.0 375.0)")
    parser.add_argument("--ds-lines-z-range", "--ds-trk-z-range", "--ds-z-range", dest="ds_lines_z_range", nargs=2, type=float, default=[375.0, 560.0], metavar=("Z1", "Z2"), help="Z extent range for reconstructed Downstream (DS) MuFilter Hough lines in cm (default: 375.0 560.0)")
    parser.add_argument("--target-region-z-range", dest="target_region_z_range", nargs=2, type=float, default=[260.0, 355.0], metavar=("Z_MIN", "Z_MAX"), help="Z boundaries for Target Region directory categorization in cm (default: 260.0 355.0)")

    args = parser.parse_args()

    # Load geometry
    geo_elements, ggeo, f_geo, scifi_module, mufilter_module, snd_geo = load_detector_geometry(args.geofile_filename)

    # Find matching files
    matched_files = sorted(glob.glob(args.input_pattern))
    if not matched_files:
        if os.path.exists(args.input_pattern):
            matched_files = [args.input_pattern]
        else:
            print(f"Error: No files found matching pattern: '{args.input_pattern}'")
            sys.exit(1)

    print("=" * 60)
    print("SND@LHC Event Display Generator (No Filtering)")
    print(f"Input Pattern         : {args.input_pattern}")
    print(f"Files Found           : {len(matched_files)}")
    print(f"Max Displays/File     : {'All Events' if args.max_displays == 0 else args.max_displays}")
    print(f"Show All MCTracks     : {args.show_all_mctracks}")
    print(f"Save Event Trees      : {args.save_trees}")
    print(f"Canvas Z Display Range: {args.canvas_z_range} cm (Start from primary muon: {args.start_from_primary_muon})")
    print(f"SciFi Lines Z Range   : {args.scifi_lines_z_range} cm")
    print(f"DS Lines Z Range      : {args.ds_lines_z_range} cm")
    print(f"Target Region Z Range : {args.target_region_z_range} cm")
    print(f"Draw Hough Lines      : {args.draw_hough}")
    print("=" * 60)

    overall_t0 = time.time()
    grand_total_events = 0
    grand_total_displays = 0
    all_sec_events = []
    created_files = []

    for idx, input_file in enumerate(matched_files, 1):
        tag = extract_file_tag(input_file)
        if "%s" in args.output_pattern:
            out_file = args.output_pattern % tag
        elif len(matched_files) > 1:
            base_name, ext = os.path.splitext(args.output_pattern)
            out_file = f"{base_name}_{tag}{ext}"
        else:
            out_file = args.output_pattern

        print(f"\n[{idx}/{len(matched_files)}] Generating Displays: '{input_file}' -> '{out_file}'")
        t_evts, n_disp, sec_list = process_single_file(
            input_file, out_file, geo_elements, ggeo, scifi_module, mufilter_module, snd_geo,
            max_displays=args.max_displays, show_all_mctracks=args.show_all_mctracks,
            draw_hough=args.draw_hough,
            canvas_z_range=tuple(args.canvas_z_range),
            start_from_primary_muon=args.start_from_primary_muon,
            scifi_lines_z_range=tuple(args.scifi_lines_z_range),
            ds_lines_z_range=tuple(args.ds_lines_z_range),
            target_region_z_range=tuple(args.target_region_z_range),
            save_trees=args.save_trees
        )

        grand_total_events += t_evts
        grand_total_displays += n_disp
        all_sec_events.extend(sec_list)
        created_files.append(out_file)

    overall_elapsed = time.time() - overall_t0

    print("\n" + "=" * 60)
    print("EVENT DISPLAY GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total Input Files Processed : {len(matched_files)}")
    print(f"Total Output Files Created   : {len(created_files)}")
    print(f"Total Events Processed      : {grand_total_events:,}")
    print(f"Total Displays Generated    : {grand_total_displays:,}")
    print(f"Secondary-Induced Events    : {len(all_sec_events):,}")
    print(f"Total Execution Time        : {overall_elapsed:.1f} s")
    print("=" * 60)

    if all_sec_events:
        print("\nList of Non-Direct / Multi-Step Trident Events:")
        for evt_num, i_evt, zv, proc_type, out_f in all_sec_events:
            print(f"  - File: {out_f} | Event #{evt_num} (Entry {i_evt}) | Process: {proc_type} | Vertex Z = {zv:.1f} cm")
        print("=" * 60)

if __name__ == "__main__":
    main()

