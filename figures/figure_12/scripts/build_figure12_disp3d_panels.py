"""Build the 8 companion 3D reconstructed-displacement-magnitude panels for
Figure 12 (Model A and D, sim 804, steps 10/46/64/68), using each model's
revised v2 BEST checkpoint. Both models share the same color range
(vmin=0/vmax=60) and camera view (elev=25, azim=-90), colored by
displacement magnitude ("disp_mag") with the viridis colormap, so the two
models' panels are directly visually comparable.

Steps 46/64/68 reuse the one intact, verified call's parameters for each
model (only step_idx/save_path differ) -- the per-step parameters for those
specific steps could not be independently recovered from the original
source, so this is disclosed as a reconstruction rather than a verbatim
per-step recovery (same caveat as for the z1-z2 panels).

Read-only with respect to models/data. Writes PNG+PDF panels into this
script's own figure_12/panels_disp3d/ directory (resolved via `__file__`);
data and checkpoint paths are plain relative paths, so run this script
from the repository root.
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")  # headless: suppress GUI popup windows from the
                        # extracted function's own plt.show() call
import matplotlib.pyplot as plt

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.dirname(SCRIPTS_DIR)  # figures/figure_12
sys.path.insert(0, SCRIPTS_DIR)

from disp3d_plot_function import plot_rollout_reconstructed_displacement_3d  # noqa: E402
from z1z2_plot_functions import rollout_single_sim_vanilla_node, rollout_single_sim_cnn_node  # noqa: E402

OUT_DIR = os.path.join(FIGURE_DIR, "panels_disp3d")
os.makedirs(OUT_DIR, exist_ok=True)

# Colorbar font enlarged (label + tick numbers) for legibility. The 3D
# axes' own tick labels are hidden by the plotting function itself
# (labelbottom/labelleft=False), so this only affects the colorbar's label
# (cbar.set_label(), drawn via axes.labelsize) and its tick numbers (drawn
# via ytick.labelsize for this vertical colorbar).
matplotlib.rcParams.update({
    "axes.labelsize": 30,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
})

NODES_CSV = "GOH_Nodes_Test_for_Visualization.csv"
ELEMS_CSV = "GOH_Elements_Test_for_Visualization.csv"
SIM_ID = 804
STEPS = [10, 46, 64, 68]

MODEL_CONFIGS = {
    # Both models share the same 0-60 color range so the panels are
    # directly comparable (an earlier convention used asymmetric per-model
    # ranges, e.g. a narrower 0-40 range for Model A).
    "A": dict(
        ckpt_path_template="checkpoints/Model_A_v2_BEST.pt",
        vmin=0.0, vmax=60.0,
        rollout_fn=rollout_single_sim_vanilla_node,
    ),
    "D": dict(
        ckpt_path_template="checkpoints/Model_D_v2_BEST.pt",
        vmin=0.0, vmax=60.0,
        rollout_fn=rollout_single_sim_cnn_node,
    ),
}

for model_name, cfg in MODEL_CONFIGS.items():
    print(f"=== Rolling out Model {model_name} (v2 BEST) for sim {SIM_ID} ===")
    rollout = cfg["rollout_fn"](
        r_modes=9, sim_id=SIM_ID, nodes_csv=NODES_CSV, elems_csv=ELEMS_CSV,
        ckpt_path_template=cfg["ckpt_path_template"],
    )
    for step in STEPS:
        for ext in ("png", "pdf"):
            save_path = os.path.join(OUT_DIR, f"fig12_disp3d_Model{model_name}_sim804_step{step}.{ext}")
            plot_rollout_reconstructed_displacement_3d(
                rollout=rollout,
                nodes_csv=NODES_CSV,
                elems_csv=ELEMS_CSV,
                step_idx=step,
                scale=1.0,
                color_by="disp_mag",
                cmap="viridis",
                show_undeformed=True,
                vmin=cfg["vmin"],
                vmax=cfg["vmax"],
                elev=25,
                azim=-90,
                save_dpi=600,
                save_path=save_path,
            )
            plt.close("all")
            print(f"  saved {save_path}")

print("DONE")
