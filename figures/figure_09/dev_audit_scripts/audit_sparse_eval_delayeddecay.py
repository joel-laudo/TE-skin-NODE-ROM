"""Archived development-time diagnostic (see ../README.md, "dev_audit_scripts/"):
exhaustively evaluate every archived checkpoint in epochs 301-500 for
the two 560-epoch delayed-decay runs (Models A, D), to check whether the
currently reported BEST checkpoint (selected only from the sparse every-5-epoch
evaluation cadence used during training in that range) is truly the global
minimum across ALL saved checkpoints, not just the ones originally evaluated.

Uses the same evaluator as training-time checkpoint selection:
campaign_fast_eval.py, invoked via subprocess (matching the training scripts'
own subprocess-isolation pattern -- calling its numba-parallel rollout in-
process alongside torch has been found to deadlock in this environment).

Not part of reproducing any reported figure. Read-only: no training, no
checkpoint modification. Depends on the private project's
campaign_fast_eval.py and per-epoch checkpoint archive, neither of which is
in this repo.
"""
import argparse
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
CAMPAIGN_EVAL = os.path.join(PROJECT_ROOT, "campaign_fast_eval.py")

MODELS = {
    "A": dict(
        rollout_fn="vanilla",
        ckpt_dir=os.path.join(PROJECT_ROOT, "FINAL_models",
                               "Model_A_ablation_fullrollout_v2_delayeddecay560", "epoch_checkpoints"),
        disp_log=os.path.join(PROJECT_ROOT, "FINAL_models",
                               "Model_A_ablation_fullrollout_v2_delayeddecay560", "disp_rmse_eval_log.txt"),
    ),
    "D": dict(
        rollout_fn="cnn",
        ckpt_dir=os.path.join(PROJECT_ROOT, "FINAL_models",
                               "Model_D_ablation_fullrollout_v2_delayeddecay560", "epoch_checkpoints"),
        disp_log=os.path.join(PROJECT_ROOT, "FINAL_models",
                               "Model_D_ablation_fullrollout_v2_delayeddecay560", "disp_rmse_eval_log.txt"),
    ),
}


def parse_training_time_log(path):
    """Return {epoch:int -> median_disp_rmse:float} from disp_rmse_eval_log.txt
    ('epoch=N median_disp_rmse=X' lines only, skip '-> new BEST' continuations)."""
    import re
    pat = re.compile(r"^epoch=(\d+)\s+median_disp_rmse=([\d.eE+\-]+)\s*$")
    out = {}
    with open(path) as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                out[int(m.group(1))] = float(m.group(2))
    return out


def run_one_eval(ckpt_path, rollout_fn, out_json_path, timeout=300):
    """Evaluate one checkpoint by launching campaign_fast_eval.py as a
    subprocess (see module docstring for why) and reading back its JSON
    result. Returns (median_disp_rmse, elapsed_seconds), or (None, elapsed)
    on timeout/failure/missing output."""
    cmd = [PY, CAMPAIGN_EVAL, "--ckpt", ckpt_path, "--rollout_fn", rollout_fn,
           "--r_modes", "9", "--out", out_json_path]
    env = dict(os.environ)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    t0 = time.time()
    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, timeout=timeout,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except subprocess.TimeoutExpired:
        return None, time.time() - t0
    if not os.path.exists(out_json_path):
        return None, time.time() - t0
    with open(out_json_path) as f:
        payload = json.load(f)
    if not payload.get("ok", False):
        return None, time.time() - t0
    return float(payload["median_disp_rmse"]), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["A", "D"])
    ap.add_argument("--epoch_start", type=int, default=301)
    ap.add_argument("--epoch_end", type=int, default=500)
    ap.add_argument("--sanity_only", action="store_true",
                     help="only re-evaluate a few already-known epochs, for cross-check")
    args = ap.parse_args()

    cfg = MODELS[args.model]
    training_time_vals = parse_training_time_log(cfg["disp_log"])
    tmp_out = os.path.join(BASE, f"_tmp_eval_{args.model}.json")
    results_path = os.path.join(BASE, f"sparse_eval_audit_{args.model}.csv")

    if args.sanity_only:
        sanity_epochs = sorted(set([300, 500]) & set(training_time_vals.keys()))
        # also add each model's own reported BEST epoch
        best_epoch = min(training_time_vals, key=training_time_vals.get)
        sanity_epochs = sorted(set(sanity_epochs) | {best_epoch})
        print(f"[{args.model}] sanity-check epochs: {sanity_epochs}")
        for ep in sanity_epochs:
            ckpt_path = os.path.join(cfg["ckpt_dir"], f"epoch_{ep:03d}.pt")
            val, dt = run_one_eval(ckpt_path, cfg["rollout_fn"], tmp_out)
            expected = training_time_vals.get(ep)
            match = "MATCH" if (val is not None and expected is not None and abs(val - expected) < 1e-4) else "CHECK"
            print(f"  epoch={ep:03d} retrospective={val} expected(training-log)={expected} [{match}] ({dt:.1f}s)")
        return

    # Full sweep: only epochs NOT already in the training-time log, within [epoch_start, epoch_end]
    already_done = set(training_time_vals.keys())
    todo = [e for e in range(args.epoch_start, args.epoch_end + 1) if e not in already_done]
    print(f"[{args.model}] {len(todo)} previously-unevaluated epochs to sweep in "
          f"[{args.epoch_start},{args.epoch_end}] (already evaluated: {len(already_done & set(range(args.epoch_start, args.epoch_end+1)))})",
          flush=True)

    rows = []
    t_start = time.time()
    for i, ep in enumerate(todo):
        ckpt_path = os.path.join(cfg["ckpt_dir"], f"epoch_{ep:03d}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  [WARN] missing checkpoint file for epoch {ep}: {ckpt_path}", flush=True)
            continue
        val, dt = run_one_eval(ckpt_path, cfg["rollout_fn"], tmp_out)
        rows.append((ep, val, dt))
        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (len(todo) - i - 1)
        print(f"  [{i+1}/{len(todo)}] epoch={ep:03d} median_disp_rmse={val} ({dt:.1f}s, elapsed={elapsed/60:.1f}min, eta={eta/60:.1f}min)",
              flush=True)

    with open(results_path, "w") as f:
        f.write("epoch,median_disp_rmse,eval_seconds,source\n")
        for ep, val, expected_val in sorted(training_time_vals.items()):
            f.write(f"{ep},{expected_val},,training_time\n")
        for ep, val, dt in rows:
            if val is not None:
                f.write(f"{ep},{val},{dt:.2f},retrospective\n")
            else:
                f.write(f"{ep},FAILED,{dt:.2f},retrospective\n")
    print(f"[{args.model}] saved {results_path}")

    if os.path.exists(tmp_out):
        os.remove(tmp_out)


if __name__ == "__main__":
    main()
