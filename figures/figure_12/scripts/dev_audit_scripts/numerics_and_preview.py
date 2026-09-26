"""Archived development-time script (see ../README.md, "dev_audit_scripts/"):
print the sim-804 quantitative comparison numbers (displacement RMSE,
latent RMSE, final Ag) for Models A and D, and build a combined preview
montage of all 8 panels -- both for internal review only, not part of the
published figure or its caption. Depends on campaign_fast_eval.py, a
private-project-only module not present in this repo.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

from z1z2_plot_functions import rollout_single_sim_vanilla_node, rollout_single_sim_cnn_node  # noqa: E402

NODES_CSV = "GOH_Nodes_Test_for_Visualization.csv"
ELEMS_CSV = "GOH_Elements_Test_for_Visualization.csv"
SIM_ID = 804

rollout_a = rollout_single_sim_vanilla_node(
    r_modes=9, sim_id=SIM_ID, nodes_csv=NODES_CSV, elems_csv=ELEMS_CSV,
    ckpt_path_template="FINAL_models/Model_A_ablation_fullrollout_v2/Model_A_Vanilla_r{r_modes}_BEST.pt",
)
rollout_d = rollout_single_sim_cnn_node(
    r_modes=9, sim_id=SIM_ID, nodes_csv=NODES_CSV, elems_csv=ELEMS_CSV,
    ckpt_path_template="FINAL_models/Model_D_ablation_fullrollout_v2/Model_D_CNN_r{r_modes}_BEST.pt",
)

from campaign_fast_eval import _compute_surface_disp_rmse_trajectory  # noqa: E402

print("=== Sim 804 quantitative comparison, revised v2 checkpoints ===")
for name, roll in [("Model A", rollout_a), ("Model D", rollout_d)]:
    u_pred = roll["u_pred_nodes"]
    u_true = roll["u_true_nodes"] if "u_true_nodes" in roll else None
    z_pred = np.asarray(roll["z_pred"])
    z_true = np.asarray(roll["z_true"])
    T = z_pred.shape[0]
    print(f"\n{name}: T={T} steps, final time t={roll['t'][-1]:.2f}")
    if u_true is not None:
        rmse_t = _compute_surface_disp_rmse_trajectory(u_pred, u_true)
        print(f"  mean disp-RMSE over trajectory: {np.mean(rmse_t):.4f} cm")
        print(f"  final-step disp-RMSE: {rmse_t[-1]:.4f} cm")
    else:
        print("  u_true_nodes not present in this rollout dict -- displacement RMSE not computed here")
    # latent (z1..z9, excluding the volume/10th state dim) RMSE
    r_modes = 9
    z_err = z_pred[:, :r_modes] - z_true[:, :r_modes]
    latent_rmse_t = np.sqrt(np.mean(z_err ** 2, axis=1))
    print(f"  mean latent (z1-z9) RMSE over trajectory: {np.mean(latent_rmse_t):.4f}")
    print(f"  final-step latent RMSE: {latent_rmse_t[-1]:.4f}")
    if "area_gain_pred_final" in roll:
        print(f"  final Ag pred: {roll['area_gain_pred_final']:.4f}")

# --- Combined preview montage ---
OUT_DIR = os.path.join(PROJECT_ROOT, "revised_figure12_v2", "panels")
PREVIEW_DIR = os.path.join(PROJECT_ROOT, "revised_figure12_v2")
STEPS = [10, 46, 64, 68]

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for col, step in enumerate(STEPS):
    imgA = mpimg.imread(os.path.join(OUT_DIR, f"fig12_ModelA_sim804_step{step}.png"))
    axes[0, col].imshow(imgA)
    axes[0, col].axis("off")
    axes[0, col].set_title(f"Model A, step {step}", fontsize=12)
    imgD = mpimg.imread(os.path.join(OUT_DIR, f"fig12_ModelD_sim804_step{step}.png"))
    axes[1, col].imshow(imgD)
    axes[1, col].axis("off")
    axes[1, col].set_title(f"Model D, step {step}", fontsize=12)
fig.tight_layout()
preview_path = os.path.join(PREVIEW_DIR, "preview_combined_grid.png")
fig.savefig(preview_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nsaved {preview_path}")
print("DONE")
