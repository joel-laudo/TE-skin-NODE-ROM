# Inference Runtime Benchmark: Model A vs. Model D

Self-contained package to benchmark per-simulation rollout inference time for
Model A (open-loop) and Model D (CNN growth-feedback), CPU-only with the
thread count explicitly pinned, over the same 185-simulation validation
split used for the manuscript's reported results. The numbers reported in
Section 4.1 were measured on Purdue's Negishi cluster (AMD EPYC 7763,
4 CPU threads pinned); this package itself is written to run on any
SLURM-based cluster.

## 1. Files already in this folder

```
benchmark_runtime.py            # main benchmark script (run this via your own job script)
rollout_benchmark_wrappers.py   # pre-loaded-model rollout wrappers (see below)
verify_bitexact.py              # Control 1: verify wrappers == authoritative functions
requirements.txt                # reference package-version pins (sanity-check only)
model_d_stabilized_eval_lib.py  # copied from project root; executable code unmodified
eval_rollout_cnn.py             # copied from project root; executable code unmodified
eval_ablation_model_a_base_closureMLP.py   # copied from project root; executable code unmodified
```

No SLURM/job-submission script is included here -- cluster job scripts
bake in site-specific account and module names, so section 5 below
documents the settings to put in your own instead of shipping one tied to
a particular cluster account.

## 2. Files/folders YOU must copy into this same folder before transferring

All paths below are relative to this folder's root (i.e. copy them so they sit
directly alongside `benchmark_runtime.py`, preserving these exact relative
paths/names -- the code uses these as hardcoded relative-path strings):

| Source (in main project) | Destination (in this package) | Size |
|---|---|---:|
| `GOH_Nodes_Test_for_Visualization.csv` | `./GOH_Nodes_Test_for_Visualization.csv` | 0.5 MB |
| `GOH_Elements_Test_for_Visualization.csv` | `./GOH_Elements_Test_for_Visualization.csv` | 0.5 MB |
| `PGOH_Ideal_tol_Tcrit_V_mu_kk1_kk2_kappa_k1_design.txt` | `./PGOH_Ideal_tol_Tcrit_V_mu_kk1_kk2_kappa_k1_design.txt` | 0.06 MB |
| `FINAL_models/Model_A_ablation_fullrollout_v2/Model_A_Vanilla_r9_BEST.pt` | `./FINAL_models/Model_A_ablation_fullrollout_v2/Model_A_Vanilla_r9_BEST.pt` | 0.1 MB |
| `FINAL_models/Model_D_ablation_fullrollout_v2/Model_D_CNN_r9_BEST.pt` | `./FINAL_models/Model_D_ablation_fullrollout_v2/Model_D_CNN_r9_BEST.pt` | 0.35 MB |
| `Design_and_Metadata.zarr/` (whole folder) | `./Design_and_Metadata.zarr/` | 0.04 MB |
| `expd_volumes.zarr/` (whole folder) | `./expd_volumes.zarr/` | 0.23 MB |
| `ip_growth_elem.zarr/` (whole folder) | `./ip_growth_elem.zarr/` | **2,321 MB** |
| `displacements.zarr/` (whole folder, OR trimmed -- see below) | `./displacements.zarr/` | 567 MB (whole) / ~1.5 MB (trimmed) |

**Total package size: ~2.9 GB with the whole `displacements.zarr`, or ~2.35 GB
with the trimmed version below.** `ip_growth_elem.zarr`'s 2.3 GB is
unavoidable -- both of its two subfolders (`snaps_SDV1`, `snaps_SDV4`) are
genuinely read by the rollout code (there's nothing else in that store to
trim). This is still a routine size for an HPC transfer (see transfer command
below), just not "tiny."

### Optional speed-up: trim `displacements.zarr`

Only `pod_full/`, `latent_displ_r9/`, and `meta/` inside `displacements.zarr`
are actually read by the rollout functions -- `snapshots/` (511 MB, full FE
displacement snapshots) is only used for diagnostic plotting, which this
benchmark never triggers. If you want to save the transfer time, copy just:

```powershell
$src = "C:\path\to\repo-root\displacements.zarr"
$dst = "C:\path\to\inference_runtime_benchmark\displacements.zarr"
New-Item -ItemType Directory -Force $dst | Out-Null
Copy-Item "$src\zarr.json" $dst
Copy-Item -Recurse "$src\pod_full" $dst
Copy-Item -Recurse "$src\latent_displ_r9" $dst
Copy-Item -Recurse "$src\meta" $dst
```

If this trimmed copy causes a zarr read error on your cluster (unlikely, but
zarr group metadata can be finicky), fall back to copying the whole
`displacements.zarr` folder instead -- `benchmark_runtime.py` doesn't care
which one you used, it just reads the same relative paths either way.

## 3. Environment

Build (or reuse) a conda/virtualenv environment containing the packages in
`requirements.txt`, then activate it before submitting -- your own job
script (see section 5) should load/activate it the way your cluster
expects (e.g. `module load anaconda` then `conda activate <your-env-name>`).

**Before submitting, run this one-line sanity check** on a login node
to confirm your environment has compatible package versions (especially
`zarr` -- its major version matters for reading the zarr stores in this
package correctly, since they were written with zarr 3.x):

```bash
python -c "import sys,torch,numpy,zarr,numba,matplotlib; print(sys.version); print('torch',torch.__version__); print('numpy',numpy.__version__); print('zarr',zarr.__version__); print('numba',numba.__version__); print('matplotlib',matplotlib.__version__)"
```

Compare against `requirements.txt` in this folder (the versions the
checkpoints/zarr stores were actually produced with: Python 3.13.9, torch
2.10.0, numpy 2.3.5, zarr 3.1.5, numba 0.62.1, matplotlib 3.10.6). Exact
matches aren't required, but a different zarr **major** version (e.g. 2.x
instead of 3.x) could fail to read these stores, so it's worth reconciling
before submitting a long run.

## 4. Transfer to your cluster

From your local machine, after placing all the files/folders from step 2:

```powershell
# Windows (PowerShell) -> cluster, via scp (adjust <username>/<cluster-host>)
scp -r "inference_runtime_benchmark" <username>@<cluster-host>:/home/<username>/
```

Given the ~2.3-2.9 GB size, `rsync` (from WSL/Git Bash) is more resumable if
the connection drops partway:

```bash
rsync -avz --progress inference_runtime_benchmark/ <username>@<cluster-host>:/home/<username>/inference_runtime_benchmark/
```

## 5. Submit the benchmark job

Write a short job script for your own cluster (SLURM or otherwise) with
these settings:

- **Resources**: 1 node, 1 task, **4 CPU cores**, ~30-minute walltime. No
  explicit memory request needed -- the job can just take the scheduler's
  default allocation for 4 CPUs (see the memory note below for why that's
  safe). Model A and Model D run together in a **single** `python`
  invocation, so one job covers both.
- **Environment**: activate the environment from section 3.
- **Before running**, set `export BENCHMARK_N_THREADS=4` -- `benchmark_runtime.py`
  takes this as the authoritative thread count rather than deriving it from
  a scheduler-reported CPU count (see section 7 for why).
- **Run**: `python benchmark_runtime.py`

A minimal SLURM example (fill in your own account/partition/module names):

```bash
#!/bin/sh -l
#SBATCH -A <your_account>
#SBATCH -p <your_cpu_partition>
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00
#SBATCH --job-name=node_rom_benchmark
#SBATCH --output=node_rom_benchmark_%j.out
#SBATCH --error=node_rom_benchmark_%j.err

module load anaconda   # or your cluster's equivalent
conda activate <your_env_name>
export BENCHMARK_N_THREADS=4
python benchmark_runtime.py
```

Submit it with `sbatch <your_script>.slurm` after transferring it alongside
the files from steps 1-2.

The no-explicit-memory-request setting above is safe because
`rollout_benchmark_wrappers.py` pre-fetches only the small per-simulation
slices each rollout actually needs, once, up front, via a single batched
zarr read (not one read per simulation, and not an eager whole-array load)
-- confirmed locally at **370 MB peak memory** for the full 185-sim,
both-model, single-process run. This prefetch step also means the timed
loop itself never touches zarr/disk -- every per-simulation call is pure
in-memory computation, satisfying the "no I/O inside the timed region"
requirement exactly.

**Before the real run**, it's worth running the bit-exact verification once
on a login node or in an interactive job (cheap, a few seconds):

```bash
python verify_bitexact.py
```

This confirms the pre-loaded-model wrapper functions reproduce the
authoritative rollout functions exactly (Control 1) -- don't trust the
benchmark numbers if this doesn't print "ALL CHECKS PASSED".

## 6. Outputs

After the job completes, check:

```
outputs/timings_modelA.csv     # sim_id, elapsed_sec, n_steps, sec_per_step -- every val sim
outputs/timings_modelD.csv     # same schema, Model D
outputs/summary.json           # mean/std/median/min/max/95% CI per model
outputs/summary.md             # ready-to-paste manuscript paragraph + table
outputs/provenance.json        # versions, hostname, CPU info, SLURM env, checkpoint paths, timestamp
node_rom_benchmark_<jobid>.out # SLURM stdout log
node_rom_benchmark_<jobid>.err # SLURM stderr log
```

`outputs/summary.md` includes a paragraph modeled on the manuscript's
existing (Model-D-only) runtime paragraph, extended to cover both models.

**What exactly each per-simulation timing includes**: the checkpoint/model
build happens once, outside all timing. Each timed call then covers that
simulation's own setup (initial condition, growth params -- now via an
in-memory dict lookup, no disk access), the full autoregressive rollout
through every step of that simulation's time horizon, and final
displacement/growth field reconstruction (stacking the per-step arrays) --
all formed in memory. No disk write of any kind happens inside a timed
interval; the CSV/JSON/summary files are all written only after every
simulation for that model has already been timed.

## 7. A note on thread pinning and Model D's growth integrator

Both models' growth integration step (`integrate_growth_matched`/
`integrate_growth_fast`, used every rollout timestep for both models) is
JIT-compiled with `numba`'s `@njit(parallel=True)` and, left unconstrained,
uses **all visible CPU cores** by default. `benchmark_runtime.py` explicitly
pins the thread count to whatever `--cpus-per-task` requests
(`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`NUMBA_NUM_THREADS`, plus `torch.set_num_threads(N)`) -- currently **4**,
matching the core count used for this project's original FE simulations --
rather than leaving it unconstrained, so the reported numbers reflect a
known, fixed CPU budget instead of "however many cores happened to be free
on the node." Both models see the same budget, so
the A-vs-D comparison stays apples-to-apples; these 4-core numbers are not
directly comparable to a 1-core run, though, if you ever want that
conservative baseline too -- just change `--cpus-per-task` back to `1` and
resubmit, no code changes needed.
