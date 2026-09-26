"""Quick 2x2 preview montage: combines the four lettered panel PNGs into a
single image, so the assembled 4-panel layout can be sanity-checked before
final figure assembly. This montage itself is not one of the figures
reported in the SI -- it's just a convenience check on panel
alignment/sizing.

Reads: panel_a_fe_nonconvergence_lettered.png,
panel_b_pod_kappa_lettered.png, panel_c_modelA_kappa_lettered.png,
panel_d_modelD_kappa_lettered.png (written by build_panel_a.py and
build_panels_bcd.py, in this same folder).

Writes: preview_2x2_montage.png.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

files = [
    "panel_a_fe_nonconvergence_lettered.png",
    "panel_b_pod_kappa_lettered.png",
    "panel_c_modelA_kappa_lettered.png",
    "panel_d_modelD_kappa_lettered.png",
]

fig, axes = plt.subplots(2, 2, figsize=(13, 12))
for ax, fname in zip(axes.flat, files):
    img = mpimg.imread(os.path.join(OUT_DIR, fname))
    ax.imshow(img)
    ax.axis("off")
fig.tight_layout()
out_path = os.path.join(OUT_DIR, "preview_2x2_montage.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"saved {out_path}")
