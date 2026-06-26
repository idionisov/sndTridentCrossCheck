import matplotlib.patches as patches
import matplotlib.pyplot as plt


def plot_trident_event(
    canvas,
    xlim=(None, None),
    ylim=(None, None),
    zlim=(None, None),
    output=None,
    figsize=(12, 10),
    conf=None,
):
    if not canvas:
        print("Canvas is empty or not found.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    axes = {1: ax1, 2: ax2}
    plotted_labels = set()

    primitives = canvas.GetListOfPrimitives()

    for obj in primitives:
        if obj.ClassName() == "TPad":
            pad_ptr = obj
            pad_id = pad_ptr.GetNumber()
            if pad_id not in axes:
                continue

            ax = axes[pad_id]
            pad_primitives = pad_ptr.GetListOfPrimitives()

            for p_obj in pad_primitives:
                cname = p_obj.ClassName()

                if "TPolyLine" in cname:
                    if (
                        p_obj.GetLineColor() == ROOT.kBlack
                        or p_obj.GetLineColor() == 0
                    ):
                        continue

                    n_points = p_obj.GetN()
                    px = []
                    py = []

                    for idx in range(n_points):
                        x_val = p_obj.GetX()[idx]
                        y_val = p_obj.GetY()[idx]
                        if x_val == 0.0 and y_val == 0.0:
                            continue
                        px.append(x_val)
                        py.append(y_val)

                    if len(px) < 3:
                        continue

                    r_color = p_obj.GetLineColor()
                    hex_color = "#BEBEBE"
                    if r_color == ROOT.kGreen - 6:
                        hex_color = "#8FBC8F"
                    elif r_color == ROOT.kRed:
                        hex_color = "#FF6347"
                    elif r_color == ROOT.kBlue + 1:
                        hex_color = "#4682B4"

                    ax.fill(
                        px,
                        py,
                        facecolor=hex_color,
                        edgecolor=hex_color,
                        alpha=0.3,
                        linewidth=1,
                        zorder=1,
                    )

                # --- Plot Hits (TGraphErrors) ---
                elif "TGraph" in cname:
                    n_points = p_obj.GetN()
                    if n_points == 0:
                        continue
                    x = [p_obj.GetPointX(idx) for idx in range(n_points)]
                    y = [p_obj.GetPointY(idx) for idx in range(n_points)]

                    name = p_obj.GetName()
                    color = "red" if "mu" in name.lower() else "blue"
                    marker = "s" if "mu" in name.lower() else "o"

                    label = name if name not in plotted_labels else ""
                    ax.scatter(
                        x, y, c=color, marker=marker, s=30, label=label, zorder=5
                    )
                    if label:
                        plotted_labels.add(label)

                # --- Plot Tracks (TLine) ---
                elif "TLine" in cname:
                    z1, z2 = p_obj.GetX1(), p_obj.GetX2()
                    coord1, coord2 = p_obj.GetY1(), p_obj.GetY2()

                    # Standard line plotting
                    ax.plot(
                        [z1, z2],
                        [coord1, coord2],
                        color="darkcyan",
                        linestyle="-",
                        alpha=0.9,
                        linewidth=2,
                        zorder=10,
                    )

                    # --- conf dictionary processing for trk_sep ---
                    if conf and "trk_sep" in conf:
                        sep_settings = conf["trk_sep"]
                        z_target = sep_settings.get("z")
                        tol = sep_settings.get("tol")

                        if (
                            isinstance(z_target, (int, float))
                            and isinstance(tol, (int, float))
                            and z2 != z1
                        ):

                            coord_at_z = coord1 + (z_target - z1) * (
                                coord2 - coord1
                            ) / (z2 - z1)

                            zmin_val = (
                                zlim[0] if zlim[0] is not None else ax.get_xlim()[0]
                            )
                            zmax_val = (
                                zlim[1] if zlim[1] is not None else ax.get_xlim()[1]
                            )
                            z_range = (
                                abs(zmax_val - zmin_val)
                                if zmin_val is not None
                                else 100
                            )

                            bar_width = z_range * 0.015
                            bar_height = tol

                            rect = patches.Rectangle(
                                (
                                    z_target - bar_width / 2.0,
                                    coord_at_z - bar_height / 2.0,
                                ),
                                bar_width,
                                bar_height,
                                linewidth=1.5,
                                edgecolor="magenta",
                                facecolor="none",
                                zorder=12,
                            )
                            ax.add_patch(rect)

                # --- Set Axis Limits (TH) ---
                elif "TH" in cname:
                    ax.set_xlim(p_obj.GetXaxis().GetXmin(), p_obj.GetXaxis().GetXmax())
                    ax.set_ylim(p_obj.GetYaxis().GetXmin(), p_obj.GetYaxis().GetXmax())

    # --- Fiducial Area Configuration Processing ---
    # Parse configurations with clean fallbacks to original defaults if settings are missing
    fid_conf = conf.get("fiducial_area", {}) if conf else {}
    
    z_fid = fid_conf.get("z")
    x_range = fid_conf.get("x_range", (-42, -11))
    y_range = fid_conf.get("y_range", (18, 49))
    draw_hlines = fid_conf.get("hlines", True) if fid_conf else True
    draw_vline = fid_conf.get("vline", False) if fid_conf else False

    # 1. Draw Horizontal Lines if configured (or active by default fallback)
    if draw_hlines:
        ax1.axhline(y=x_range[0], color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="Target Range")
        ax1.axhline(y=x_range[1], color="red", linestyle="--", linewidth=1.5, alpha=0.8)
        
        ax2.axhline(y=y_range[0], color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="Target Range")
        ax2.axhline(y=y_range[1], color="red", linestyle="--", linewidth=1.5, alpha=0.8)

    # 2. Draw Vertical Lines with End Caps indicating location and physical dimension span
    if draw_vline and isinstance(z_fid, (int, float)):
        # Calculate the geometric center and the half-length error span for matplotlib errorbars
        x_center = sum(x_range) / 2.0
        x_err = abs(x_range[1] - x_range[0]) / 2.0
        
        y_center = sum(y_range) / 2.0
        y_err = abs(y_range[1] - y_range[0]) / 2.0

        # ax1 (XZ Projection): Horizontal placement is z_fid, Vertical span is x_range
        ax1.errorbar(
            z_fid, x_center, yerr=x_err, fmt='none', ecolor='red', 
            elinewidth=2, capsize=6, capthick=2, zorder=11, label="Fiducial Plane"
        )
        
        # ax2 (YZ Projection): Horizontal placement is z_fid, Vertical span is y_range
        ax2.errorbar(
            z_fid, y_center, yerr=y_err, fmt='none', ecolor='red', 
            elinewidth=2, capsize=6, capthick=2, zorder=11
        )

    ax1.set_ylabel("X [cm]")
    ax1.set_ylim(xlim)
    ax1.set_xlim(zlim)
    ax2.set_ylabel("Y [cm]")
    ax2.set_ylim(ylim)
    ax2.set_xlim(zlim)
    ax2.set_xlabel("Z [cm]")

    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(
        by_label.values(), by_label.keys(), loc="upper right", fontsize="small"
    )

    fig.suptitle(canvas.GetTitle())
    plt.tight_layout()
    ax1.grid()
    ax2.grid()
    if output is not None:
        plt.savefig(output)
    plt.show()