"""Archived development-time script (see ../README.md, "dev_audit_scripts/"):
one-time extraction of the z1-z2 latent-slice plotting functions (Model A
and Model D), verbatim, from Visualize_Latent_Trajectories_extracted.py (a
private, notebook-converted script that does not exist in this repo), with
only the ckpt_path_template defaults repointed at the revised v2 BEST
checkpoints. Its output, z1z2_plot_functions.py, is now the checked-in,
directly maintained artifact -- this extraction script is not re-run.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(PROJECT_ROOT, "Visualize_Latent_Trajectories_extracted.py")

with open(SRC, encoding="utf-8") as f:
    lines = f.readlines()

# 1-based line numbers. Each function's end is its own closing `return {...}`
# brace (line 1705 / 4476) -- not the next top-level def, which is actually
# an unrelated set_pub_plot_style() helper defined in a later, separate
# notebook cell. The count_top_level_defs check below guards against a
# naive "next def" boundary accidentally sweeping that helper in too.
model_a_func = "".join(lines[1278 - 1:1705])
model_d_func = "".join(lines[4018 - 1:4476])

def count_top_level_defs(src):
    return sum(1 for line in src.splitlines() if line.startswith("def "))

assert "def plot_selected_sim_vanilla_node_z1_z2_slice(" in model_a_func
assert count_top_level_defs(model_a_func) == 1, (
    f"Model A extraction captured {count_top_level_defs(model_a_func)} top-level defs, expected 1")
assert "def plot_selected_sim_cnn_node_z1_z2_slice(" in model_d_func
assert count_top_level_defs(model_d_func) == 1, (
    f"Model D extraction captured {count_top_level_defs(model_d_func)} top-level defs, expected 1")

OLD_A_PATH = 'ckpt_path_template: str = "FINAL_models/Model_A/Model_A_Vanilla_r{r_modes}_BEST.pt",'
NEW_A_PATH = 'ckpt_path_template: str = "FINAL_models/Model_A_ablation_fullrollout_v2/Model_A_Vanilla_r{r_modes}_BEST.pt",'
assert model_a_func.count(OLD_A_PATH) == 1
model_a_func = model_a_func.replace(OLD_A_PATH, NEW_A_PATH)

OLD_D_PATH = 'ckpt_path_template: str = "FINAL_models/Model_D/Model_D_CNN_r{r_modes}_BEST.pt",'
NEW_D_PATH = 'ckpt_path_template: str = "FINAL_models/Model_D_ablation_fullrollout_v2/Model_D_CNN_r{r_modes}_BEST.pt",'
assert model_d_func.count(OLD_D_PATH) == 1
model_d_func = model_d_func.replace(OLD_D_PATH, NEW_D_PATH)

header = '''"""Z1-Z2 latent phase-portrait plotting functions for Models A and D --
EXTRACTED VERBATIM from "Visualize Latent Trajectories.ipynb" (original
pre-v2 notebook), with ONLY the ckpt_path_template defaults repointed at the
revised v2 BEST checkpoints:
  Model A: FINAL_models/Model_A_ablation_fullrollout_v2/Model_A_Vanilla_r9_BEST.pt (epoch 356)
  Model D: FINAL_models/Model_D_ablation_fullrollout_v2/Model_D_CNN_r9_BEST.pt (epoch 357)

All plotting/styling logic (streamplot, trajectory overlay, markers, axes,
figure size, dpi) is preserved character-for-character from the original
source -- no redesign. build_vanilla_node_from_ckpt/build_cnn_node_from_ckpt/
load_raw_data/rollout_single_sim_vanilla_node/rollout_single_sim_cnn_node_matched
are supplied by the already-validated v2-era eval modules
(eval_ablation_model_a_base_closureMLP.py, eval_rollout_cnn.py) rather than
the notebook's own local (potentially pre-v2-specific) copies, since those
modules are confirmed compatible with the v2 checkpoint format throughout
this project's session history.
"""
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend -- must be set before any pyplot import
import matplotlib.pyplot as plt

from eval_ablation_model_a_base_closureMLP import (
    build_vanilla_node_from_ckpt,
    load_raw_data,
    rollout_single_sim_vanilla_node,
)
from eval_rollout_cnn import rollout_single_sim_cnn_node_matched as rollout_single_sim_cnn_node
from model_d_stabilized_eval_lib import build_cnn_node_from_ckpt

import matplotlib as mpl


def set_pub_plot_style():
    """Verbatim from the original notebook (defined + called there right
    before the sim-804 plotting calls) -- global rcParams applied to every
    saved panel in the original Figure 12 source."""
    mpl.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.5,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })


'''

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "z1z2_plot_functions.py")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(header)
    f.write(model_a_func)
    f.write("\n\n")
    f.write(model_d_func)

print(f"wrote {out_path}")
print(f"Model A function: {len(model_a_func.splitlines())} lines")
print(f"Model D function: {len(model_d_func.splitlines())} lines")
