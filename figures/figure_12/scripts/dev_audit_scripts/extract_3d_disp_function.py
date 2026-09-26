"""Archived development-time script (see ../README.md, "dev_audit_scripts/"):
one-time extraction of the 3D reconstructed-displacement-magnitude plotting
function, verbatim, from Visualize_Latent_Trajectories_extracted.py (a
private, notebook-converted script that does not exist in this repo). Its
Model A section (lines 1987-2218) and Model D section (lines 4616-4847)
were byte-identical -- a single, model-agnostic function (depends only on
rollout["u_pred_nodes"], not on any model-specific internals), so it needs
no repointing to v2 checkpoints itself; the v2-ness comes entirely from
which rollout dict is passed in by the caller. Its output,
disp3d_plot_function.py, is now the checked-in, directly maintained
artifact -- this extraction script is not re-run.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(PROJECT_ROOT, "Visualize_Latent_Trajectories_extracted.py")

with open(SRC, encoding="utf-8") as f:
    lines = f.readlines()

func = "".join(lines[1987 - 1:2219])
assert "def plot_rollout_reconstructed_displacement_3d(" in func
assert sum(1 for line in func.splitlines() if line.startswith("def ")) == 1

header = '''"""3D reconstructed-displacement-magnitude plotting function -- EXTRACTED
VERBATIM from "Visualize Latent Trajectories.ipynb" (confirmed byte-identical
between its Model A and Model D notebook sections). No changes needed for
v2 compatibility: this function only reads rollout["u_pred_nodes"] (T,N,3)
and the shared bottom-surface mesh, neither of which are model-specific.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend -- must be set before any pyplot import
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from eval_ablation_model_a_base_closureMLP import load_bottom_surface_mesh_direct


'''

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "disp3d_plot_function.py")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(header)
    f.write(func)

print(f"wrote {out_path} ({len(func.splitlines())} lines extracted)")
