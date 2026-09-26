"""Per-simulation rollout inference-time benchmark for Model A vs. Model D.

Times a full autoregressive rollout (all timesteps of one simulation, start
to finish) for each of the 185 validation simulations, for both the
open-loop Model A and the CNN growth-feedback Model D. This produces the
per-simulation inference-time statistics (mean/median/CI) reported in
Section 4.1 as the runtime cost of Model D's growth-feedback CNN pathway
relative to Model A.

Usage:
    python benchmark_runtime.py                  # both models, all 185 val sims
    python benchmark_runtime.py --models A       # Model A only
    python benchmark_runtime.py --max-sims 5     # dry run on first 5 val sims

Design:
  - Thread-pinning env vars are set BEFORE importing torch/numba/numpy, so
    both torch's intra-op thread pool and numba's parallel (@njit(parallel=True))
    growth integrator honor a single, explicit CPU-count budget instead of
    silently fanning out across every visible core.
  - Checkpoint load + model build happens ONCE per model, before any timing.
  - A single untimed warm-up rollout (on the first validation sim) triggers
    numba JIT compilation and populates all lru_cache/module-global data
    caches (POD basis, mesh, zarr metadata) so the *timed* loop measures
    steady-state per-simulation cost only. The warm-up sim is still included
    in the real timed loop afterward -- no sim is skipped.
  - Per-sim timing brackets exactly: growth-parameter lookup for that sim,
    initial-condition setup, the full autoregressive rollout loop, and final
    field reconstruction/packing -- i.e. everything the user asked for
    ("complete per-simulation rollout inference, including any normal
    model/setup operations required for prediction and reconstruction").
  - All CSV/JSON writes happen strictly AFTER each model's timed loop
    finishes -- disk-writing time is never inside a timed interval.
"""
import os

# Deliberately not derived solely from SLURM_CPUS_PER_TASK: that env var is
# not authoritative on every cluster's accounting/billing setup, so it is
# only used as a fallback for ad hoc/manual runs outside a job script
# (e.g. local testing). BENCHMARK_N_THREADS is set explicitly by the job
# script (see README section 5) and takes precedence when present; our own
# process never spawns more OS threads than this value, regardless of how
# many cores the job's cgroup reports as available.
_N_THREADS = os.environ.get("BENCHMARK_N_THREADS") or os.environ.get("SLURM_CPUS_PER_TASK", "1")
os.environ["OMP_NUM_THREADS"] = _N_THREADS
os.environ["MKL_NUM_THREADS"] = _N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = _N_THREADS
os.environ["NUMBA_NUM_THREADS"] = _N_THREADS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU-only unless caller overrides

import argparse
import json
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

torch.set_num_threads(int(_N_THREADS))

import eval_ablation_model_a_base_closureMLP as ma  # noqa: E402
import eval_rollout_cnn as ed  # noqa: E402
from rollout_benchmark_wrappers import (  # noqa: E402
    rollout_A_preloaded, rollout_D_preloaded, prefetch_growth_data,
)

R_MODES = 9
NODES_CSV = "GOH_Nodes_Test_for_Visualization.csv"
ELEMS_CSV = "GOH_Elements_Test_for_Visualization.csv"
A_CKPT = "FINAL_models/Model_A_ablation_fullrollout_v2/Model_A_Vanilla_r9_BEST.pt"
D_CKPT = "FINAL_models/Model_D_ablation_fullrollout_v2/Model_D_CNN_r9_BEST.pt"
OUT_DIR = os.path.join(SCRIPT_DIR, "outputs")


def sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gather_provenance():
    prov = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "cpu_processor": platform.processor(),
        "os_cpu_count": os.cpu_count(),
        "n_threads_requested": int(_N_THREADS),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_cpus_on_node": os.environ.get("SLURM_CPUS_ON_NODE"),
        "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_job_account": os.environ.get("SLURM_JOB_ACCOUNT"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "checkpoint_A": A_CKPT,
        "checkpoint_D": D_CKPT,
    }
    try:
        import zarr
        prov["zarr_version"] = zarr.__version__
    except Exception as e:
        prov["zarr_version"] = f"unavailable ({e})"
    try:
        import numba
        prov["numba_version"] = numba.__version__
    except Exception as e:
        prov["numba_version"] = f"unavailable ({e})"
    try:
        import matplotlib
        prov["matplotlib_version"] = matplotlib.__version__
    except Exception as e:
        prov["matplotlib_version"] = f"unavailable ({e})"

    # Linux CPU model name, if available (more informative than platform.processor())
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    prov["cpu_model_name"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SCRIPT_DIR, stderr=subprocess.STDOUT
        ).decode().strip()
        prov["git_commit"] = commit
    except Exception:
        prov["git_commit"] = "not a git repository (or git unavailable)"

    return prov


def bootstrap_mean_ci(x, n_resamples=5000, seed=0):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    rng = np.random.default_rng(seed)
    boot = np.array([np.mean(x[rng.integers(0, n, n)]) for _ in range(n_resamples)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def sem_normal_ci(x):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    sem = np.std(x, ddof=1) / np.sqrt(n)
    mean = np.mean(x)
    return float(mean - 1.96 * sem), float(mean + 1.96 * sem)


def summarize(times_sec):
    x = np.asarray(times_sec, dtype=np.float64)
    boot_lo, boot_hi = bootstrap_mean_ci(x)
    sem_lo, sem_hi = sem_normal_ci(x)
    return {
        "n": int(len(x)),
        "mean_sec": float(np.mean(x)),
        "std_sec": float(np.std(x, ddof=1)),
        "median_sec": float(np.median(x)),
        "min_sec": float(np.min(x)),
        "max_sec": float(np.max(x)),
        "ci95_mean_bootstrap": [boot_lo, boot_hi],
        "ci95_mean_sem_normal": [sem_lo, sem_hi],
    }


def run_model_benchmark(name, val_sims, ckpt_path, build_fn, wrapper_fn, growth_cache):
    print(f"=== {name}: loading checkpoint (once) ===")
    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_fn(ckpt, device)
    model.eval()

    print(f"=== {name}: warm-up rollout (untimed, sim {val_sims[0]}) ===")
    _ = wrapper_fn(
        model=model, ckpt=ckpt, device=device, r_modes=R_MODES, sim_id=int(val_sims[0]),
        nodes_csv=NODES_CSV, elems_csv=ELEMS_CSV, growth_cache=growth_cache,
    )

    print(f"=== {name}: timed loop over {len(val_sims)} validation sims ===")
    rows = []
    for i, sim_id in enumerate(val_sims):
        t0 = time.perf_counter()
        roll = wrapper_fn(
            model=model, ckpt=ckpt, device=device, r_modes=R_MODES, sim_id=int(sim_id),
            nodes_csv=NODES_CSV, elems_csv=ELEMS_CSV, growth_cache=growth_cache,
        )
        sync_if_cuda()
        t1 = time.perf_counter()

        elapsed = t1 - t0
        n_steps = int(roll["n_steps"])
        rows.append({
            "sim_id": int(sim_id),
            "elapsed_sec": elapsed,
            "n_steps": n_steps,
            "sec_per_step": elapsed / n_steps if n_steps > 0 else float("nan"),
        })
        if (i + 1) % 25 == 0 or (i + 1) == len(val_sims):
            print(f"  {i + 1}/{len(val_sims)} sims timed (last: sim {sim_id}, {elapsed:.4f}s, {n_steps} steps)")

    return rows


def write_csv(path, rows):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sim_id", "elapsed_sec", "n_steps", "sec_per_step"])
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(path, provenance, summaries):
    lines = []
    lines.append("# Runtime Benchmark Summary\n")
    lines.append(f"Run on `{provenance['hostname']}`, {provenance['timestamp_utc']}, "
                  f"SLURM job {provenance.get('slurm_job_id')}, "
                  f"{provenance['n_threads_requested']} CPU thread(s) pinned "
                  f"(`OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`NUMBA_NUM_THREADS`).\n")
    lines.append("| Model | n sims | mean (s) | std (s) | median (s) | min (s) | max (s) | 95% CI (bootstrap) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for name, s in summaries.items():
        ci = s["ci95_mean_bootstrap"]
        lines.append(
            f"| {name} | {s['n']} | {s['mean_sec']:.4f} | {s['std_sec']:.4f} | "
            f"{s['median_sec']:.4f} | {s['min_sec']:.4f} | {s['max_sec']:.4f} | "
            f"[{ci[0]:.4f}, {ci[1]:.4f}] |"
        )
    lines.append("")
    n_threads = provenance.get("n_threads_requested", "?")
    thread_phrase = "single-threaded" if n_threads == 1 else f"pinned to {n_threads} CPU threads"
    lines.append("## Suggested manuscript text\n")
    for name, s in summaries.items():
        lines.append(
            f"{name}: mean per-simulation rollout inference time of "
            f"{s['mean_sec']:.3f} s (SD {s['std_sec']:.3f} s, median {s['median_sec']:.3f} s, "
            f"range [{s['min_sec']:.3f}, {s['max_sec']:.3f}] s, 95% CI of the mean "
            f"[{s['ci95_mean_bootstrap'][0]:.3f}, {s['ci95_mean_bootstrap'][1]:.3f}] s, "
            f"n={s['n']} validation simulations), measured {thread_phrase} on "
            f"{provenance.get('cpu_model_name', provenance['cpu_processor'])} "
            f"(Purdue Negishi cluster).\n"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["A", "D"], choices=["A", "D"])
    parser.add_argument("--max-sims", type=int, default=None, help="limit to first N val sims (dry-run)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    provenance = gather_provenance()
    with open(os.path.join(OUT_DIR, "provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)
    print("provenance:", json.dumps(provenance, indent=2))

    val_sims = np.asarray(ma.val_sims, dtype=np.int64)
    if args.max_sims is not None:
        val_sims = val_sims[: args.max_sims]

    # Pre-fetch every val sim's growth params + t=0 growth initial condition
    # ONCE, up front (untimed setup, same spirit as loading the checkpoint
    # once) -- see prefetch_growth_data's docstring in
    # rollout_benchmark_wrappers.py. This is what makes it safe to run both
    # models in a single process without running out of memory: neither
    # model's rollout ever eagerly loads the full ~2.3 GB growth-snapshot
    # arrays, and the timed loop below never touches zarr/disk at all.
    # Shared across both models since the underlying physical data is
    # model-independent.
    print("=== pre-fetching per-sim growth data (untimed) ===")
    (_, _, sim_index, time_vals, _, _, _) = ma.load_raw_data(R_MODES)
    growth_cache = prefetch_growth_data(val_sims, sim_index, time_vals)

    # Merge with any existing summary.json rather than overwrite it -- lets
    # --models A and --models D be run as separate invocations if ever
    # useful (e.g. re-running just one model), while a single combined
    # invocation (the default, and what the README section 5 job script
    # example uses) writes both in one pass.
    summary_path = os.path.join(OUT_DIR, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summaries = json.load(f)
    else:
        summaries = {}

    if "A" in args.models:
        rows_a = run_model_benchmark(
            "Model A", val_sims, A_CKPT, ma.build_vanilla_node_from_ckpt, rollout_A_preloaded,
            growth_cache,
        )
        write_csv(os.path.join(OUT_DIR, "timings_modelA.csv"), rows_a)
        summaries["Model A"] = summarize([r["elapsed_sec"] for r in rows_a])

    if "D" in args.models:
        rows_d = run_model_benchmark(
            "Model D", val_sims, D_CKPT, ed.build_cnn_node_from_ckpt, rollout_D_preloaded,
            growth_cache,
        )
        write_csv(os.path.join(OUT_DIR, "timings_modelD.csv"), rows_d)
        summaries["Model D"] = summarize([r["elapsed_sec"] for r in rows_d])

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)
    write_summary_md(os.path.join(OUT_DIR, "summary.md"), provenance, summaries)

    print()
    print("=== DONE ===")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
