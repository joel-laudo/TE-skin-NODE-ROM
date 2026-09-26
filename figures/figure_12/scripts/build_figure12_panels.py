"""Build Figure 12's individual z1-z2 latent phase-portrait panels (Model A
and Model D) using each model's revised v2 BEST checkpoint, for sim 804 at
rollout-truncation steps 10/46/64/68.

Sim 804 is used for both Model A and Model D panels. The call parameters
for step 10 are the intact, verified parameters recovered from the original
source for both models. Steps 46/64/68 reuse those same parameters (only
traj_end_step/save_path vary) as the best-available faithful
reconstruction -- the original source's state for those specific steps
could not be fully recovered, so this is disclosed as a reconstruction
rather than an exact re-derivation.

Read-only with respect to models/data. Writes PNG+PDF panels into this
script's own figure_12/panels/ directory (resolved via `__file__`); data
and checkpoint paths (e.g. nodes_csv, checkpoints/Model_{A,D}_v2_BEST.pt)
are plain relative paths, so run this script from the repository root.
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")  # headless: suppress GUI popup windows
import matplotlib.pyplot as plt

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.dirname(SCRIPTS_DIR)  # figures/figure_12
sys.path.insert(0, SCRIPTS_DIR)

from z1z2_plot_functions import (  # noqa: E402
    set_pub_plot_style,
    plot_selected_sim_vanilla_node_z1_z2_slice,
    plot_selected_sim_cnn_node_z1_z2_slice,
    rollout_single_sim_vanilla_node,
    rollout_single_sim_cnn_node,
)

OUT_DIR = os.path.join(FIGURE_DIR, "panels")
os.makedirs(OUT_DIR, exist_ok=True)

set_pub_plot_style()
# These panels get shrunk down when placed into the composite Figure 12, so
# font sizes here are bumped above set_pub_plot_style()'s defaults to stay
# legible after scaling down. Applied AFTER set_pub_plot_style() so these
# overrides aren't clobbered by it.
matplotlib.rcParams.update({
    "font.size": 24,
    "axes.labelsize": 32,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
})

NODES_CSV = "GOH_Nodes_Test_for_Visualization.csv"
ELEMS_CSV = "GOH_Elements_Test_for_Visualization.csv"
SIM_ID = 804
STEPS = [10, 46, 64, 68]
Z1_LIM = (-520, 720)
Z2_LIM = (-150, 90)

print("=== Rolling out Model A (v2 BEST, epoch 356) for sim 804 ===")
rollout_a = rollout_single_sim_vanilla_node(
    r_modes=9, sim_id=SIM_ID, nodes_csv=NODES_CSV, elems_csv=ELEMS_CSV,
    ckpt_path_template="checkpoints/Model_A_v2_BEST.pt",
)
print(f"  rollout keys: {list(rollout_a.keys())}, T={len(rollout_a['t'])}")

print("=== Rolling out Model D (v2 BEST, epoch 357) for sim 804 ===")
rollout_d = rollout_single_sim_cnn_node(
    r_modes=9, sim_id=SIM_ID, nodes_csv=NODES_CSV, elems_csv=ELEMS_CSV,
    ckpt_path_template="checkpoints/Model_D_v2_BEST.pt",
)
print(f"  rollout keys: {list(rollout_d.keys())}, T={len(rollout_d['t'])}")

for step in STEPS:
    print(f"\n--- Model A, sim {SIM_ID}, traj_end_step={step} ---")
    save_path = os.path.join(OUT_DIR, f"fig12_ModelA_sim804_step{step}.png")
    meta = plot_selected_sim_vanilla_node_z1_z2_slice(
        r_modes=9, sim_id=SIM_ID, rollout=rollout_a, traj_end_step=step,
        grid_n=25, normalize_arrows=False, use_streamplot=True,
        z1_lim=Z1_LIM, z2_lim=Z2_LIM,
        quiver_color="#b22222", quiver_alpha=0.85, quiver_width=0.0025,
        quiver_headwidth=4.0, quiver_headlength=5.0, quiver_headaxislength=4.5,
        quiver_linewidth=0.4, arrow_length_frac=0.6,
        # Terminal-tangent (red) arrow shown on every step except step 68
        # (turned off there, since step 68 is the final/terminal step and
        # the tangent arrow has nothing further to point toward).
        show_terminal_tangent=(step != 68), show_true_traj=True,
        # Terminal-tangent (red) arrow enlarged beyond this plotting
        # function's defaults (width=0.004/headwidth=4.5/headlength=6.0/
        # headaxislength=5.0/length_frac=2.2) for visibility at this panel size.
        terminal_tangent_width=0.008, terminal_tangent_headwidth=7.0,
        terminal_tangent_headlength=9.0, terminal_tangent_headaxislength=7.5,
        terminal_tangent_length_frac=3.0,
        # Red terminal "x" marker enlarged on the step-68 panels specifically
        # (the only step where it renders, since show_terminal_x_only_if_last
        # defaults to True); default size is end_markersize=55.
        end_markersize=(300 if step == 68 else 55),
        end_marker_linewidth=(8 if step == 68 else None),
        pred_linewidth=3.8, true_color="crimson", true_linewidth=3.8, true_alpha=0.85, title=None,
        save_path=save_path, save_dpi=600,
    )
    # also emit a vector PDF at the same styling
    save_path_pdf = os.path.join(OUT_DIR, f"fig12_ModelA_sim804_step{step}.pdf")
    plot_selected_sim_vanilla_node_z1_z2_slice(
        r_modes=9, sim_id=SIM_ID, rollout=rollout_a, traj_end_step=step,
        grid_n=25, normalize_arrows=False, use_streamplot=True,
        z1_lim=Z1_LIM, z2_lim=Z2_LIM,
        quiver_color="#b22222", quiver_alpha=0.85, quiver_width=0.0025,
        quiver_headwidth=4.0, quiver_headlength=5.0, quiver_headaxislength=4.5,
        quiver_linewidth=0.4, arrow_length_frac=0.6,
        show_terminal_tangent=(step != 68), show_true_traj=True,
        terminal_tangent_width=0.008, terminal_tangent_headwidth=7.0,
        terminal_tangent_headlength=9.0, terminal_tangent_headaxislength=7.5,
        terminal_tangent_length_frac=3.0,
        end_markersize=(300 if step == 68 else 55),
        end_marker_linewidth=(8 if step == 68 else None),
        pred_linewidth=3.8, true_color="crimson", true_linewidth=3.8, true_alpha=0.85, title=None,
        save_path=save_path_pdf, save_dpi=600,
    )
    plt.close("all")
    print(f"  saved {save_path}")
    print(f"  saved {save_path_pdf}")

for step in STEPS:
    print(f"\n--- Model D, sim {SIM_ID}, traj_end_step={step} ---")
    save_path = os.path.join(OUT_DIR, f"fig12_ModelD_sim804_step{step}.png")
    plot_selected_sim_cnn_node_z1_z2_slice(
        r_modes=9, sim_id=SIM_ID, rollout=rollout_d, traj_end_step=step,
        grid_n=25, normalize_arrows=False, use_streamplot=True,
        z1_lim=Z1_LIM, z2_lim=Z2_LIM,
        quiver_color="black", quiver_alpha=0.85, quiver_width=0.0035,
        quiver_headwidth=4.0, quiver_headlength=5.0, quiver_headaxislength=4.5,
        quiver_linewidth=0.6, arrow_length_frac=0.6,
        # Terminal-tangent (red) arrow shown for Model D too, on every step
        # except step 68 -- same enlarged sizing as Model A's arrow above.
        show_terminal_tangent=(step != 68), show_true_traj=True,
        terminal_tangent_width=0.008, terminal_tangent_headwidth=7.0,
        terminal_tangent_headlength=9.0, terminal_tangent_headaxislength=7.5,
        terminal_tangent_length_frac=3.0,
        end_markersize=(300 if step == 68 else 55),
        end_marker_linewidth=(8 if step == 68 else None),
        pred_linewidth=3.8, true_linewidth=3.8, true_alpha=0.85, title=None,
        growth_key="g_pred_grid",
        save_path=save_path, save_dpi=600,
    )
    save_path_pdf = os.path.join(OUT_DIR, f"fig12_ModelD_sim804_step{step}.pdf")
    plot_selected_sim_cnn_node_z1_z2_slice(
        r_modes=9, sim_id=SIM_ID, rollout=rollout_d, traj_end_step=step,
        grid_n=25, normalize_arrows=False, use_streamplot=True,
        z1_lim=Z1_LIM, z2_lim=Z2_LIM,
        quiver_color="black", quiver_alpha=0.85, quiver_width=0.0035,
        quiver_headwidth=4.0, quiver_headlength=5.0, quiver_headaxislength=4.5,
        quiver_linewidth=0.6, arrow_length_frac=0.6,
        show_terminal_tangent=(step != 68), show_true_traj=True,
        terminal_tangent_width=0.008, terminal_tangent_headwidth=7.0,
        terminal_tangent_headlength=9.0, terminal_tangent_headaxislength=7.5,
        terminal_tangent_length_frac=3.0,
        end_markersize=(300 if step == 68 else 55),
        end_marker_linewidth=(8 if step == 68 else None),
        pred_linewidth=3.8, true_linewidth=3.8, true_alpha=0.85, title=None,
        growth_key="g_pred_grid",
        save_path=save_path_pdf, save_dpi=600,
    )
    plt.close("all")
    print(f"  saved {save_path}")
    print(f"  saved {save_path_pdf}")

print("\nDONE")
