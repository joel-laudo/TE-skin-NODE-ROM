"""3D reconstructed-displacement-magnitude plotting function for Figure 12's
companion disp3d panels, called by build_figure12_disp3d_panels.py. This
function is shared between Model A and Model D: it only needs
rollout["u_pred_nodes"] (T, N, 3) and the shared bottom-surface mesh,
neither of which is model-specific, so the same code renders both models'
panels.

Loads the mesh via `evaluation/eval_model_a.py`'s
`load_bottom_surface_mesh_direct` (imported below with `evaluation/` added
to `sys.path` relative to this file's own location, so this resolves
regardless of the caller's working directory).
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend -- must be set before any pyplot import
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# figures/figure_12/scripts -> figures/figure_12 -> figures -> repository root
_EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "evaluation")
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from eval_model_a import load_bottom_surface_mesh_direct


def plot_rollout_reconstructed_displacement_3d(
    rollout: dict,
    nodes_csv: str,
    elems_csv: str,
    step_idx: int | None = None,
    scale: float = 1.0,
    color_by: str = "disp_mag",   # "disp_mag", "ux", "uy", "uz"
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    show_undeformed: bool = True,
    undeformed_alpha: float = 0.18,
    deformed_alpha: float = 1.0,
    edgecolor: str | None = None,
    linewidth: float = 0.0,
    elev: float = 22,
    azim: float = -60,
    figsize=(9, 7),
    title: str | None = None,
    axis_equal: bool = True,
    show_colorbar: bool = True,
    save_path: str | None = None,
    save_dpi: int = 400,
):
    """
    Plot reconstructed predicted displacement on the bottom-surface mesh
    for a specified rollout timepoint.

    The undeformed mesh is drawn as a faint translucent gray surface (if
    show_undeformed); the deformed mesh (undeformed node positions plus
    `scale` times the predicted displacement) is drawn on top, shaded by
    `color_by` (displacement magnitude by default) using `cmap`, with a
    colorbar giving the scale. Face colors are the average of their
    corner nodes' scalar values (Poly3DCollection colors per-face, not
    per-vertex).

    Parameters
    ----------
    rollout : dict
        Output from rollout_single_sim_vanilla_node(...), must contain:
          - "u_pred_nodes": (T, N, 3)
          - "t": (T,)
          - "sim_id"
    nodes_csv, elems_csv : str
        Mesh files used by load_bottom_surface_mesh_direct.
    step_idx : int or None
        Rollout time index to plot. Defaults to final step.
    scale : float
        Visual scaling applied to displacement before plotting.
    color_by : str
        One of: "disp_mag", "ux", "uy", "uz".
    vmin, vmax : float or None
        Fixed color limits. If None, inferred from the selected frame.
    """

    if "u_pred_nodes" not in rollout:
        raise KeyError("rollout must contain 'u_pred_nodes'.")

    u_pred_nodes = np.asarray(rollout["u_pred_nodes"], dtype=np.float64)
    t_arr = np.asarray(rollout.get("t", []), dtype=np.float64)

    if u_pred_nodes.ndim != 3 or u_pred_nodes.shape[2] != 3:
        raise ValueError(f"Expected u_pred_nodes shape (T, N, 3), got {u_pred_nodes.shape}")

    T, N, _ = u_pred_nodes.shape
    if step_idx is None:
        step_idx = T - 1
    step_idx = int(np.clip(step_idx, 0, T - 1))

    # Load bottom surface mesh
    node_xyz_bot, elem_conn_bot, node_ids_bot, elem_node_ids_bot = load_bottom_surface_mesh_direct(
        nodes_csv, elems_csv
    )

    node_xyz_bot = np.asarray(node_xyz_bot, dtype=np.float64)
    elem_conn_bot = np.asarray(elem_conn_bot)

    if node_xyz_bot.shape[0] != N:
        raise ValueError(
            f"Mesh node count ({node_xyz_bot.shape[0]}) does not match rollout u_pred_nodes ({N})."
        )

    u = u_pred_nodes[step_idx]
    x_def = node_xyz_bot + scale * u

    if color_by == "disp_mag":
        node_scalar = np.linalg.norm(u, axis=1)
        cbar_label = r"$\|\mathbf{u}\|$"
    elif color_by == "ux":
        node_scalar = u[:, 0]
        cbar_label = r"$u_x$"
    elif color_by == "uy":
        node_scalar = u[:, 1]
        cbar_label = r"$u_y$"
    elif color_by == "uz":
        node_scalar = u[:, 2]
        cbar_label = r"$u_z$"
    else:
        raise ValueError("color_by must be one of: 'disp_mag', 'ux', 'uy', 'uz'")

    # Build polygon faces and per-face scalar
    faces_def = []
    faces_und = []
    face_scalar = []

    for conn in elem_conn_bot:
        conn = np.asarray(conn, dtype=int)

        pts_def = x_def[conn]
        pts_und = node_xyz_bot[conn]

        faces_def.append(pts_def)
        faces_und.append(pts_und)

        # Average nodal scalar to face color
        face_scalar.append(np.mean(node_scalar[conn]))

    face_scalar = np.asarray(face_scalar, dtype=np.float64)

    # Fixed or automatic color limits
    cmin = np.min(face_scalar) if vmin is None else float(vmin)
    cmax = np.max(face_scalar) if vmax is None else float(vmax)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if show_undeformed:
        poly_und = Poly3DCollection(
            faces_und,
            facecolor=(0.7, 0.7, 0.7, undeformed_alpha),
            edgecolor="none",
            linewidth=0.0,
        )
        ax.add_collection3d(poly_und)

    poly_def = Poly3DCollection(
        faces_def,
        linewidth=linewidth,
        edgecolor=edgecolor if edgecolor is not None else "none",
        alpha=deformed_alpha,
    )
    poly_def.set_array(face_scalar)
    poly_def.set_cmap(cmap)
    poly_def.set_clim(cmin, cmax)
    ax.add_collection3d(poly_def)

    # Set axes limits from both undeformed and deformed geometry
    xyz_all = np.vstack([node_xyz_bot, x_def])
    xmin, ymin, zmin = xyz_all.min(axis=0)
    xmax, ymax, zmax = xyz_all.max(axis=0)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    if axis_equal:
        ranges = np.array([xmax - xmin, ymax - ymin, zmax - zmin], dtype=float)
        max_range = np.max(ranges)
        xmid = 0.5 * (xmin + xmax)
        ymid = 0.5 * (ymin + ymax)
        zmid = 0.5 * (zmin + zmax)

        ax.set_xlim(xmid - 0.5 * max_range, xmid + 0.5 * max_range)
        ax.set_ylim(ymid - 0.5 * max_range, ymid + 0.5 * max_range)
        ax.set_zlim(0, zmid + 0.5 * max_range)

    ax.view_init(elev=elev, azim=azim)

    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    zticks = ax.get_zticks()

    # hide the default z-axis line
    ax.zaxis.line.set_color((0, 0, 0, 0))

    # Draw a custom z-axis line, but only up to ztop (matching the z-tick
    # range set below) rather than the full data range.
    z0, z1 = ax.get_zlim()
    ztop = 80   # highest z-tick shown on the visible axis

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # Corner of the bounding box where matplotlib draws the z-axis by default
    x_axis_corner = xmax
    y_axis_corner = ymax

    ax.plot(
        [x_axis_corner, x_axis_corner],
        [y_axis_corner, y_axis_corner],
        [z0, ztop],
        color='k',
        lw=1.2
    )

    ax.set_xticks([0, 100, 200])
    ax.set_yticks([0, 100, 200])
    ax.set_zticks([0, 80])
    ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(axis='y', labelleft=False)
    ax.tick_params(axis='z', labelleft=False)
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')

    sim_id = rollout.get("sim_id", "unknown")
    t_val = t_arr[step_idx] if len(t_arr) > step_idx else step_idx

    if title is None:
        title = f"Predicted reconstructed displacement | sim={sim_id} | step={step_idx} | t={t_val:.4g}"
    ax.set_title(title)

    if show_colorbar:
        mappable = plt.cm.ScalarMappable(cmap=cmap)
        mappable.set_array(face_scalar)
        mappable.set_clim(cmin, cmax)
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.75, pad=0.08)
        cbar.set_label(cbar_label)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight")
        print(f"[saved figure] {save_path}")

    plt.show()

    return {
        "step_idx": step_idx,
        "t": float(t_val) if np.isscalar(t_val) else t_val,
        "u": u,
        "x_def": x_def,
        "node_scalar": node_scalar,
        "face_scalar": face_scalar,
        "vmin_used": cmin,
        "vmax_used": cmax,
    }

